import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from chainlit.context import ChainlitContextException
from chainlit.context import context
from chainlit.data.base import BaseDataLayer
from chainlit.step import StepDict
from chainlit.types import Feedback, PageInfo, PaginatedResponse, Pagination, ThreadDict, ThreadFilter
from chainlit.user import PersistedUser, User


def _utcnow() -> str:
    return datetime.utcnow().isoformat()


def _build_thread_name_from_text(text: str) -> str:
    short_text = (text or "").strip().replace("\n", " ")
    if not short_text:
        return "Yeni sohbet"
    if len(short_text) > 42:
        short_text = short_text[:39] + "..."
    return short_text


class LocalDataLayer(BaseDataLayer):
    def __init__(self, root_dir: Path):
        self.root_dir = root_dir
        self.threads_dir = self.root_dir / "threads"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.threads_dir.mkdir(parents=True, exist_ok=True)
        self.users_file = self.root_dir / "users.json"
        self.feedbacks_file = self.root_dir / "feedbacks.json"

    def _read_json(self, path: Path, default):
        if not path.exists():
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _write_json(self, path: Path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_users(self) -> Dict[str, Dict]:
        return self._read_json(self.users_file, {})

    def _save_users(self, users: Dict[str, Dict]):
        self._write_json(self.users_file, users)

    def _thread_path(self, thread_id: str) -> Path:
        return self.threads_dir / f"{thread_id}.json"

    def _load_thread(self, thread_id: str) -> Optional[ThreadDict]:
        thread = self._read_json(self._thread_path(thread_id), None)
        if not thread:
            return None
        return self._normalize_thread(thread)

    def _save_thread(self, thread: ThreadDict):
        self._write_json(self._thread_path(thread["id"]), self._normalize_thread(thread))

    def _current_user(self):
        try:
            return getattr(context.session, "user", None)
        except ChainlitContextException:
            return None

    def _normalize_thread(self, thread: ThreadDict) -> ThreadDict:
        thread["steps"] = sorted(
            thread.get("steps") or [],
            key=lambda step: step.get("createdAt") or step.get("start") or "",
        )
        thread["elements"] = thread.get("elements") or []
        thread["tags"] = thread.get("tags") or []
        thread["metadata"] = thread.get("metadata") or {}
        thread["createdAt"] = thread.get("createdAt") or _utcnow()

        if not thread.get("name"):
            for step in thread["steps"]:
                if step.get("type") == "user_message":
                    thread["name"] = _build_thread_name_from_text(
                        step.get("output") or step.get("input") or ""
                    )
                    break

        if not thread.get("name"):
            created_at = thread.get("createdAt", "")
            thread["name"] = f"Yeni sohbet {created_at[:16].replace('T', ' ')}".strip()

        thread["metadata"]["updatedAt"] = thread["metadata"].get("updatedAt") or thread["createdAt"]
        return thread

    def _ensure_thread(self, thread_id: str) -> ThreadDict:
        existing = self._load_thread(thread_id)
        if existing:
            return existing

        user = self._current_user()
        created_at = _utcnow()
        thread: ThreadDict = {
            "id": thread_id,
            "createdAt": created_at,
            "name": None,
            "userId": getattr(user, "id", None),
            "userIdentifier": getattr(user, "identifier", None),
            "tags": [],
            "metadata": {"updatedAt": created_at},
            "steps": [],
            "elements": [],
        }
        self._save_thread(thread)
        return thread

    async def get_user(self, identifier: str) -> Optional[PersistedUser]:
        users = self._load_users()
        user = users.get(identifier)
        if not user:
            return None
        return PersistedUser(**user)

    async def create_user(self, user: User) -> Optional[PersistedUser]:
        users = self._load_users()
        existing = users.get(user.identifier)

        if existing:
            # Her girişte metadata'yı güncelle (role, can_access_shared vb.)
            merged_meta = {**(existing.get("metadata") or {}), **(user.metadata or {})}
            existing["metadata"] = merged_meta
            existing["display_name"] = user.display_name
            users[user.identifier] = existing
            self._save_users(users)
            return PersistedUser(**existing)

        persisted = {
            "id": str(uuid.uuid4()),
            "identifier": user.identifier,
            "display_name": user.display_name,
            "metadata": user.metadata or {},
            "createdAt": _utcnow(),
        }
        users[user.identifier] = persisted
        self._save_users(users)
        return PersistedUser(**persisted)

    async def delete_feedback(self, feedback_id: str) -> bool:
        feedbacks = self._read_json(self.feedbacks_file, {})
        existed = feedback_id in feedbacks
        if existed:
            feedbacks.pop(feedback_id, None)
            self._write_json(self.feedbacks_file, feedbacks)
        return existed

    async def upsert_feedback(self, feedback: Feedback) -> str:
        feedbacks = self._read_json(self.feedbacks_file, {})
        feedback_id = feedback.id or str(uuid.uuid4())
        feedbacks[feedback_id] = {
            "id": feedback_id,
            "forId": feedback.forId,
            "threadId": feedback.threadId,
            "value": feedback.value,
            "comment": feedback.comment,
        }
        self._write_json(self.feedbacks_file, feedbacks)
        return feedback_id

    async def create_element(self, element):
        thread_id = getattr(element, "thread_id", None) or context.session.thread_id
        thread = self._ensure_thread(thread_id)
        element_dict = element.to_dict()
        thread["elements"] = [e for e in (thread.get("elements") or []) if e.get("id") != element_dict.get("id")]
        thread["elements"].append(element_dict)
        thread["metadata"] = thread.get("metadata") or {}
        thread["metadata"]["updatedAt"] = _utcnow()
        self._save_thread(thread)

    async def get_element(self, thread_id: str, element_id: str):
        thread = self._load_thread(thread_id)
        if not thread:
            return None
        for element in thread.get("elements") or []:
            if element.get("id") == element_id:
                return element
        return None

    async def delete_element(self, element_id: str, thread_id: Optional[str] = None):
        if not thread_id:
            return
        thread = self._load_thread(thread_id)
        if not thread:
            return
        thread["elements"] = [e for e in (thread.get("elements") or []) if e.get("id") != element_id]
        thread["metadata"] = thread.get("metadata") or {}
        thread["metadata"]["updatedAt"] = _utcnow()
        self._save_thread(thread)

    async def create_step(self, step_dict: StepDict):
        thread = self._ensure_thread(step_dict["threadId"])
        thread["steps"] = [s for s in thread.get("steps", []) if s.get("id") != step_dict.get("id")]
        thread["steps"].append(step_dict)
        if step_dict.get("type") == "user_message" and not thread.get("name"):
            thread["name"] = _build_thread_name_from_text(
                step_dict.get("output") or step_dict.get("input") or ""
            )
        thread["metadata"] = thread.get("metadata") or {}
        thread["metadata"]["updatedAt"] = step_dict.get("createdAt") or _utcnow()
        self._save_thread(thread)

    async def update_step(self, step_dict: StepDict):
        await self.create_step(step_dict)

    async def delete_step(self, step_id: str):
        for path in self.threads_dir.glob("*.json"):
            thread = self._read_json(path, None)
            if not thread:
                continue
            original_count = len(thread.get("steps", []))
            thread["steps"] = [s for s in thread.get("steps", []) if s.get("id") != step_id]
            if len(thread["steps"]) != original_count:
                thread["metadata"] = thread.get("metadata") or {}
                thread["metadata"]["updatedAt"] = _utcnow()
                self._write_json(path, thread)
                return

    async def get_thread_author(self, thread_id: str) -> str:
        thread = self._load_thread(thread_id)
        return thread.get("userIdentifier") if thread else ""

    async def delete_thread(self, thread_id: str):
        path = self._thread_path(thread_id)
        if path.exists():
            path.unlink()

    async def list_threads(self, pagination: Pagination, filters: ThreadFilter) -> PaginatedResponse[ThreadDict]:
        threads: List[ThreadDict] = []
        current_user = self._current_user()
        current_user_id = getattr(current_user, "id", None)

        for path in self.threads_dir.glob("*.json"):
            thread = self._read_json(path, None)
            if not thread:
                continue
            thread = self._normalize_thread(thread)

            filter_user_id = filters.userId or current_user_id
            if filter_user_id and thread.get("userId") != filter_user_id:
                continue

            search = (filters.search or "").strip().lower()
            if search:
                haystacks = [
                    thread.get("name") or "",
                    " ".join((step.get("output") or "") for step in thread.get("steps", [])),
                ]
                if not any(search in value.lower() for value in haystacks):
                    continue

            threads.append(thread)

        threads.sort(
            key=lambda thread: (thread.get("metadata") or {}).get("updatedAt", thread.get("createdAt", "")),
            reverse=True,
        )

        start_index = 0
        if pagination.cursor:
            for index, thread in enumerate(threads):
                if thread["id"] == pagination.cursor:
                    start_index = index + 1
                    break

        data = threads[start_index : start_index + pagination.first]
        end_cursor = data[-1]["id"] if data else None
        has_next_page = start_index + pagination.first < len(threads)

        return PaginatedResponse(
            pageInfo=PageInfo(
                hasNextPage=has_next_page,
                startCursor=data[0]["id"] if data else None,
                endCursor=end_cursor,
            ),
            data=data,
        )

    async def get_thread(self, thread_id: str) -> Optional[ThreadDict]:
        return self._load_thread(thread_id)

    async def update_thread(
        self,
        thread_id: str,
        name: Optional[str] = None,
        user_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
        tags: Optional[List[str]] = None,
    ):
        thread = self._ensure_thread(thread_id)
        if name is not None:
            thread["name"] = name
        if user_id is not None:
            thread["userId"] = user_id
            users = self._load_users()
            for user in users.values():
                if user.get("id") == user_id:
                    thread["userIdentifier"] = user.get("identifier")
                    break
        if metadata is not None:
            merged_metadata = thread.get("metadata") or {}
            merged_metadata.update(metadata)
            thread["metadata"] = merged_metadata
        if tags is not None:
            thread["tags"] = tags
        thread["metadata"] = thread.get("metadata") or {}
        thread["metadata"]["updatedAt"] = _utcnow()
        self._save_thread(thread)

    async def build_debug_url(self) -> str:
        return ""

    async def close(self) -> None:
        return None

    async def get_favorite_steps(self, user_id: str) -> List[StepDict]:
        favorites: List[StepDict] = []
        for path in self.threads_dir.glob("*.json"):
            thread = self._read_json(path, None)
            if not thread or thread.get("userId") != user_id:
                continue
            for step in thread.get("steps", []):
                if (step.get("metadata") or {}).get("favorite"):
                    favorites.append(step)
        return favorites
