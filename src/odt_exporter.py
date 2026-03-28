"""
ODT (OpenDocument Text) export modülü - Markdown desteğiyle.
Soru-cevap konuşmalarını .odt formatında kaydeder, markdown formatını korur.
"""

import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from odf.opendocument import OpenDocumentText
from odf.text import P, H, Span, List as ODFList, ListItem, LineBreak
from odf.style import Style, TextProperties, ParagraphProperties, ListLevelProperties
from odf.table import Table, TableColumn, TableRow, TableCell, TableHeaderRows
from odf import teletype


class ODTExporter:
    """Konuşmaları ODT formatında export eder - Markdown desteğiyle"""
    
    def __init__(self, export_dir: str = "exports"):
        """
        Args:
            export_dir: Export dosyalarının kaydedileceği klasör
        """
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(exist_ok=True)
    
    def create_styles(self, doc: OpenDocumentText):
        """ODT için stil tanımlamaları oluştur"""
        # Başlık stili
        title_style = Style(name="Title", family="paragraph")
        title_style.addElement(TextProperties(fontsize="18pt", fontweight="bold"))
        title_style.addElement(ParagraphProperties(textalign="center"))
        doc.styles.addElement(title_style)
        
        # Soru stili (kalın, mavi)
        question_style = Style(name="Question", family="text")
        question_style.addElement(TextProperties(fontsize="12pt", fontweight="bold", color="#2563EB"))
        doc.styles.addElement(question_style)
        
        # Cevap stili (normal)
        answer_style = Style(name="Answer", family="text")
        answer_style.addElement(TextProperties(fontsize="11pt", color="#1F2937"))
        doc.styles.addElement(answer_style)
        
        # Kalın metin (markdown **)
        bold_style = Style(name="Bold", family="text")
        bold_style.addElement(TextProperties(fontweight="bold"))
        doc.styles.addElement(bold_style)

        # İtalik metin (markdown *)
        italic_style = Style(name="Italic", family="text")
        italic_style.addElement(TextProperties(fontstyle="italic"))
        doc.styles.addElement(italic_style)
        
        # Kod/monospace (markdown ``)
        code_style = Style(name="Code", family="text")
        code_style.addElement(TextProperties(
            fontfamily="monospace",
            fontsize="10pt",
            color="#DC2626"
        ))
        doc.styles.addElement(code_style)
        
        # Başlık 2
        heading2_style = Style(name="Heading2", family="paragraph")
        heading2_style.addElement(TextProperties(fontsize="14pt", fontweight="bold", color="#1E40AF"))
        heading2_style.addElement(ParagraphProperties(marginbottom="0.2cm"))
        doc.styles.addElement(heading2_style)
        
        # Başlık 3
        heading3_style = Style(name="Heading3", family="paragraph")
        heading3_style.addElement(TextProperties(fontsize="12pt", fontweight="bold", color="#3B82F6"))
        doc.styles.addElement(heading3_style)
        
        # Kaynak stili (italik, küçük)
        source_style = Style(name="Source", family="text")
        source_style.addElement(TextProperties(fontsize="10pt", fontstyle="italic", color="#6B7280"))
        doc.styles.addElement(source_style)
        
        # Ayraç stili
        separator_style = Style(name="Separator", family="paragraph")
        separator_style.addElement(ParagraphProperties(textalign="center"))
        separator_style.addElement(TextProperties(color="#9CA3AF"))
        doc.styles.addElement(separator_style)
        
        # Liste stili
        list_style = Style(name="ListItem", family="paragraph")
        list_style.addElement(ParagraphProperties(marginleft="0.5cm"))
        doc.styles.addElement(list_style)
        
        # Tablo stilleri
        # Tablo stili (genel tablo)
        table_style = Style(name="Table", family="table")
        table_style.addElement(TextProperties(fontsize="10pt"))
        doc.automaticstyles.addElement(table_style)
        
        # Tablo hücre stili (normal)
        table_cell_style = Style(name="TableCell", family="table-cell")
        table_cell_style.addElement(ParagraphProperties(margin="0.05cm"))
        table_cell_style.addElement(TextProperties(fontsize="10pt"))
        doc.automaticstyles.addElement(table_cell_style)
        
        # Tablo başlık hücre stili (kalın, arkaplan)
        table_header_cell_style = Style(name="TableHeaderCell", family="table-cell")
        table_header_cell_style.addElement(ParagraphProperties(margin="0.05cm", backgroundcolor="#E5E7EB"))
        table_header_cell_style.addElement(TextProperties(fontsize="10pt", fontweight="bold"))
        doc.automaticstyles.addElement(table_header_cell_style)
        
        # Tablo sütun stili (genişlik)
        table_column_style = Style(name="TableColumn", family="table-column")
        table_column_style.addElement(TextProperties(fontsize="10pt"))
        doc.automaticstyles.addElement(table_column_style)
        
        # Tablo başlığı stili (tablo üst başlığı)
        table_title_style = Style(name="TableTitle", family="paragraph")
        table_title_style.addElement(TextProperties(fontsize="12pt", fontweight="bold", color="#1E40AF"))
        table_title_style.addElement(ParagraphProperties(margintop="0.3cm", marginbottom="0.1cm"))
        doc.styles.addElement(table_title_style)
        
        # Tablo not stili (tablo alt notları)
        table_note_style = Style(name="TableNote", family="paragraph")
        table_note_style.addElement(TextProperties(fontsize="9pt", fontstyle="italic", color="#6B7280"))
        table_note_style.addElement(ParagraphProperties(margintop="0.1cm", marginbottom="0.3cm"))
        doc.styles.addElement(table_note_style)
    
    # LaTeX → Unicode dönüşüm tablosu
    _LATEX_MAP = {
        r'\pm': '±', r'\mu': 'μ', r'\times': '×', r'\div': '÷',
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ',
        r'\epsilon': 'ε', r'\theta': 'θ', r'\lambda': 'λ', r'\pi': 'π',
        r'\sigma': 'σ', r'\phi': 'φ', r'\omega': 'ω', r'\infty': '∞',
        r'\leq': '≤', r'\geq': '≥', r'\neq': '≠', r'\approx': '≈',
        r'\cdot': '·', r'\degree': '°', r'\circ': '°', r'\Delta': 'Δ',
        r'\Sigma': 'Σ', r'\sqrt': '√', r'\rightarrow': '→', r'\leftarrow': '←',
        r'\uparrow': '↑', r'\downarrow': '↓', r'\ldots': '…',
    }

    def _convert_latex(self, text: str) -> str:
        r"""$...$ veya \(...\) içindeki LaTeX ifadelerini Unicode'a çevir."""
        def _replace_expr(expr: str) -> str:
            for latex, uni in self._LATEX_MAP.items():
                expr = expr.replace(latex, uni)
            # Tanınmayan \komutları temizle
            expr = re.sub(r'\\[a-zA-Z]+', '', expr)
            # Birden fazla boşluğu tek boşluğa indir
            expr = re.sub(r' {2,}', ' ', expr)
            return expr.strip()

        # $expr$ kalıplarını dönüştür
        text = re.sub(r'\$([^$]+)\$', lambda m: _replace_expr(m.group(1)), text)
        # \( expr \) kalıplarını dönüştür
        text = re.sub(r'\\\((.+?)\\\)', lambda m: _replace_expr(m.group(1)), text)
        return text

    def _make_bold_span(self, text: str) -> Span:
        """ODT görüntüleyicilerinde güvenilir çalışan kalın span oluştur."""
        span = Span(stylename="Bold", text=text)
        return span

    def _render_inline_markdown(self, text: str, para: P) -> None:
        """
        Inline markdown span'larını parse edip verilen paragrafa ekler.
        Desteklenen: **bold**, *italic*, `code`, <br>, $LaTeX$
        """
        # Önce LaTeX math ifadelerini Unicode'a çevir
        text = self._convert_latex(text)

        remaining = text
        while remaining:
            # <br> — gerçek satır sonu
            if remaining.startswith('<br>'):
                para.addElement(LineBreak())
                remaining = remaining[4:]
                continue

            # **bold**
            bold_match = re.match(r'\*\*(.+?)\*\*', remaining)
            if bold_match:
                para.addElement(self._make_bold_span(bold_match.group(1)))
                remaining = remaining[bold_match.end():]
                continue

            # `code`
            code_match = re.match(r'`([^`]+)`', remaining)
            if code_match:
                para.addElement(Span(stylename="Code", text=code_match.group(1)))
                remaining = remaining[code_match.end():]
                continue

            # *italic* — ** ile başlamıyorsa (bold ile karışmasın)
            if not remaining.startswith('**'):
                italic_match = re.match(r'\*([^\*]+?)\*', remaining)
                if italic_match:
                    para.addElement(Span(stylename="Italic", text=italic_match.group(1)))
                    remaining = remaining[italic_match.end():]
                    continue

            # Sonraki özel karaktere kadar normal metin
            next_special = len(remaining)
            for pattern in [r'\*\*', r'\*', r'`', r'<br>']:
                match = re.search(pattern, remaining)
                if match and match.start() < next_special:
                    next_special = match.start()

            if next_special == 0:
                para.addText(remaining[0])
                remaining = remaining[1:]
            elif next_special < len(remaining):
                para.addText(remaining[:next_special])
                remaining = remaining[next_special:]
            else:
                para.addText(remaining)
                break

    def parse_markdown_line(self, line: str, doc: OpenDocumentText, context: Dict = None) -> Optional[Any]:
        """
        Markdown satırını parse et ve uygun ODT elementi döndür.
        Desteklenen: ## başlıklar, **yıldızlı başlık**, inline **bold**/*italic*/`code`

        Args:
            line: Parse edilecek satır
            doc: ODT dökümanı (stiller için)
            context: Tablo parsing durumu gibi context bilgisi

        Returns:
            ODT element (P, H, Table) veya None
        """
        stripped_line = line.strip()

        # Tablo satırı — ayrıca işlenecek
        if stripped_line.startswith('|') and stripped_line.endswith('|'):
            return None

        # Başlık kontrolü (###, ##, #)
        if stripped_line.startswith("###"):
            return H(outlinelevel=3, stylename="Heading3", text=stripped_line[3:].strip())
        if stripped_line.startswith("##"):
            return H(outlinelevel=2, stylename="Heading2", text=stripped_line[2:].strip())
        if stripped_line.startswith("#"):
            return H(outlinelevel=2, stylename="Heading2", text=stripped_line[1:].strip())

        # Satırın tamamı **text** ise başlık olarak yorumla
        if stripped_line.startswith('**') and stripped_line.endswith('**'):
            heading_text = stripped_line[2:-2].strip()
            if heading_text and '**' not in heading_text and len(heading_text) > 2:
                return H(outlinelevel=2, stylename="Heading2", text=heading_text)

        # Liste öğesi: *, -, + veya 1. 2. gibi numaralı liste
        list_match = re.match(r'^([\*\-\+])\s{1,4}(.+)$', stripped_line)
        if not list_match:
            list_match = re.match(r'^\d+[\.\)]\s+(.+)$', stripped_line)
            if list_match:
                content = list_match.group(1)
                para = P(stylename="ListItem")
                para.addText("• ")
                self._render_inline_markdown(content, para)
                return para
        if list_match:
            content = list_match.group(2)
            para = P(stylename="ListItem")
            para.addText("• ")
            self._render_inline_markdown(content, para)
            return para

        # Normal paragraf — inline markdown render
        para = P()
        self._render_inline_markdown(stripped_line, para)
        return para
    
    def parse_markdown_table(self, lines: List[str], start_idx: int) -> Tuple[Optional[Table], int]:
        """
        Markdown tablosunu parse et ve ODT Table oluştur
        
        Args:
            lines: Tüm satırlar listesi
            start_idx: Tablonun başladığı index
            
        Returns:
            (Table object, son işlenen satır indexi) veya (None, start_idx) tablo değilse
        """
        # Tablo başlığı kontrol et (en az 3 satır gerekli: başlık, separator, veri)
        if start_idx + 2 >= len(lines):
            return None, start_idx
        
        # İlk satır tablo satırı mı?
        first_line = lines[start_idx].strip()
        if not (first_line.startswith('|') and first_line.endswith('|')):
            return None, start_idx
        
        # İkinci satır separator mı? (|-|-|-| veya |:---|:---:|---:|)
        second_line = lines[start_idx + 1].strip()
        if not (second_line.startswith('|') and second_line.endswith('|')):
            return None, start_idx
        
        # Tablo satırlarını topla
        table_rows = []
        current_idx = start_idx
        
        while current_idx < len(lines):
            line = lines[current_idx].strip()
            # Tablo satırı değilse dur
            if not (line.startswith('|') and line.endswith('|')):
                # Boş satır veya separator satırı kontrol et
                if line == '' or re.match(r'^\|[-:\s|]+\|$', line):
                    # Bu hala tablo parçası olabilir, devam et
                    pass
                else:
                    break
            
            table_rows.append(line)
            current_idx += 1
        
        # En az 2 satır olmalı (başlık + separator)
        if len(table_rows) < 2:
            return None, start_idx
        
        # Tabloyu parse et
        parsed_rows = []
        for row in table_rows:
            # Satırdaki hücreleri ayır (baştaki ve sondaki | karakterlerini çıkar)
            cells = [cell.strip() for cell in row[1:-1].split('|')]
            parsed_rows.append(cells)
        
        # Tablo oluştur
        table = Table(stylename="Table")
        
        # Sütun sayısını belirle (ilk satırdaki hücre sayısı)
        num_columns = len(parsed_rows[0]) if parsed_rows else 0
        for _ in range(num_columns):
            col = TableColumn(stylename="TableColumn")
            table.addElement(col)
        
        # Başlık satırını ekle
        if len(parsed_rows) > 0:
            header_row = TableRow()
            for cell_text in parsed_rows[0]:
                cell = TableCell(stylename="TableHeaderCell")
                para = P()
                # Hücre içindeki markdown formatını parse et (basit bold/italic)
                self._add_formatted_text(cell_text, para)
                cell.addElement(para)
                header_row.addElement(cell)
            table.addElement(header_row)
        
        # Veri satırlarını ekle (separator hariç)
        for i in range(2, len(parsed_rows)):
            # Separator satırını atla (|-|-|-| formatı)
            if re.match(r'^[-:\s|]+$', ''.join(parsed_rows[i])):
                continue
            
            data_row = TableRow()
            for cell_text in parsed_rows[i]:
                cell = TableCell(stylename="TableCell")
                para = P()
                # Hücre içindeki markdown formatını parse et
                self._add_formatted_text(cell_text, para)
                cell.addElement(para)
                data_row.addElement(cell)
            table.addElement(data_row)
        
        return table, current_idx
    
    def _add_formatted_text(self, text: str, para: P) -> None:
        """
        Tablo hücresi metnini formatla ve paragrafa ekle.
        Boş hücre için açık boşluk garantisi; içerik için _render_inline_markdown'a delege eder.

        Args:
            text: Formatlanacak metin
            para: ODT paragrafı
        """
        if not text:
            para.addText("")
            return
        self._render_inline_markdown(text, para)
    
    def _generate_filename(self, question: str, answer: str, timestamp: str) -> str:
        """
        Soru ve cevaptan andıran dosya adı oluştur.
        Format: {tarih}_{anahtar_kelime1}_{anahtar_kelime2}.odt
        
        Args:
            question: Soru metni
            answer: Cevap metni
            timestamp: Zaman damgası (YYYYMMDD_HHMMSS formatında)
            
        Returns:
            Dosya adı (otomatik oluşturulmuş)
        """
        # Türkçe karakter dönüşümü
        def turkce_ayikla(text: str) -> str:
            """Türkçe karakterleri İngilizce'ye çevir ve sadece harf/rakam bırak"""
            turkce_map = {
                'ş': 's', 'Ş': 'S',
                'ğ': 'g', 'Ğ': 'G',
                'ü': 'u', 'Ü': 'U',
                'ö': 'o', 'Ö': 'O',
                'ç': 'c', 'Ç': 'C',
                'ı': 'i', 'İ': 'I'
            }
            result = ""
            for char in text:
                if char in turkce_map:
                    result += turkce_map[char]
                elif char.isalnum() or char.isspace():
                    result += char
            return result
        
        # Sorudan anahtar kelimeleri al (en uzun 3 kelime)
        question_clean = turkce_ayikla(question.lower())
        question_words = [w for w in question_clean.split() if len(w) > 2][:3]
        
        # Cevaptan anahtar kelimeleri al (en uzun 2 kelime)
        answer_clean = turkce_ayikla(answer.lower())
        answer_words = [w for w in answer_clean.split() if len(w) > 2][:2]
        
        # Anahtar kelimeleri birleştir
        keywords = question_words + answer_words
        if not keywords:
            keywords = ["konuşma"]
        
        # Dosya adı oluştur: tarih_anahtar1_anahtar2
        filename = f"{timestamp}_{'_'.join(keywords)}.odt"
        
        # Maksimum uzunluk (255 karakter - dosya sistemi limiti)
        if len(filename) > 200:  # Güvenlik payı
            filename = filename[:200]
        
        return filename
    
    def export_conversation(
        self,
        qa_pairs: List[Dict[str, Any]],
        filename: str = None,
        model_info: Dict[str, Any] = None,
        performance_metrics: Dict[str, Any] = None
    ) -> Path:
        """
        Konuşmayı ODT formatında export et.
        
        Args:
            qa_pairs: Soru-cevap çiftleri listesi
                     Her öğe: {"question": str, "answer": str, "sources": List[dict], "timestamp": str,
                               "mode": str, "model_info": Dict, "performance_metrics": Dict}
            filename: Dosya adı (None ise otomatik oluşturulur)
            model_info: Model bilgileri {"model_id": str, "model_name": str, "loaded_models": List}
            performance_metrics: Performans metrikleri {"generation_time": float, "token_usage": Dict,
                                                        "agentic_metrics": Dict (opsiyonel)}
            
        Returns:
            Path: Kaydedilen dosyanın yolu
        """
        if not qa_pairs:
            raise ValueError("Kaydedilecek konuşma yok!")
        
        # Dosya adı oluştur
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # İlk soruyu al (tüm konuşmayı temsil edecek)
            first_question = qa_pairs[0].get("question", "")
            first_answer = qa_pairs[0].get("answer", "")
            filename = self._generate_filename(first_question, first_answer, timestamp)
        
        if not filename.endswith('.odt'):
            filename += '.odt'
        
        filepath = self.export_dir / filename
        
        # ODT dökümanı oluştur
        doc = OpenDocumentText()
        self.create_styles(doc)
        
        # Başlık
        title = H(outlinelevel=1, stylename="Title", text="RAG Chat Konuşması")
        doc.text.addElement(title)
        
        # Metadata
        doc.text.addElement(P(text=""))
        metadata = P(text=f"Tarih: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
        doc.text.addElement(metadata)
        
        total_qa = P(text=f"Toplam Soru-Cevap: {len(qa_pairs)}")
        doc.text.addElement(total_qa)
        
        # Model bilgisi
        if model_info:
            model_name = model_info.get("model_name", "Bilinmeyen Model")
            model_id = model_info.get("model_id", "bilinmeyen")
            model_p = P(text=f"Model: {model_name} ({model_id})")
            doc.text.addElement(model_p)
        
        # Performans metrikleri
        if performance_metrics:
            doc.text.addElement(P(text=""))
            doc.text.addElement(P(text="═══════════════════════════════════════"))
            perf_title = P()
            perf_title.addElement(Span(stylename="Question", text="PERFORMANS METRİKLERİ"))
            doc.text.addElement(perf_title)
            doc.text.addElement(P(text="═══════════════════════════════════════"))
            
            # Çalışma modu
            if performance_metrics.get("mode"):
                mode = performance_metrics["mode"].upper()
                mode_text = "🚀 Simple" if mode == "SIMPLE" else "🧠 Agentic"
                doc.text.addElement(P(text=f"Çalışma Modu: {mode_text}"))
            
            # Üretim süresi
            if performance_metrics.get("generation_time"):
                gen_time = performance_metrics["generation_time"]
                doc.text.addElement(P(text=f"Üretim Süresi: {gen_time:.2f} saniye"))
            
            # Token kullanımı
            if performance_metrics.get("token_usage"):
                token_usage = performance_metrics["token_usage"]
                input_tokens = token_usage.get("input_tokens") or token_usage.get("total_input_tokens", 0)
                output_tokens = token_usage.get("output_tokens") or token_usage.get("total_output_tokens", 0)
                total_tokens = token_usage.get("total_tokens") or (input_tokens + output_tokens)
                
                doc.text.addElement(P(text=f"Token Kullanımı: {input_tokens:,} / {output_tokens:,} (Giriş/Çıkış)"))
                doc.text.addElement(P(text=f"Toplam Token: {total_tokens:,}"))
            
            # Agentic metrikleri
            if performance_metrics.get("agentic_metrics"):
                agent_metrics = performance_metrics["agentic_metrics"]
                if agent_metrics.get("iteration_count"):
                    doc.text.addElement(P(text=f"İterasyon Sayısı: {agent_metrics['iteration_count']}"))
                if agent_metrics.get("search_count"):
                    doc.text.addElement(P(text=f"Arama Sayısı: {agent_metrics['search_count']}"))
                if agent_metrics.get("thought_count"):
                    doc.text.addElement(P(text=f"Düşünme Sayısı: {agent_metrics['thought_count']}"))
                if agent_metrics.get("visual_search_count"):
                    doc.text.addElement(
                        P(text=f"Görsel arama (ColPali): {agent_metrics['visual_search_count']}")
                    )
        
        doc.text.addElement(P(text="═" * 60))
        doc.text.addElement(P(text=""))
        
        # Her soru-cevap çiftini ekle
        for idx, qa in enumerate(qa_pairs, 1):
            # Soru numarası
            qa_header = P(text=f"─── Soru {idx} ───")
            doc.text.addElement(qa_header)
            doc.text.addElement(P(text=""))
            
            # Zaman damgası ve mod bilgisi
            meta_text = f"⏰ {qa.get('timestamp', 'N/A')}"
            if qa.get("mode"):
                mode_emoji = "🚀" if qa["mode"] == "simple" else "🧠"
                meta_text += f" | {mode_emoji} Mod: {qa['mode'].upper()}"
            
            time_p = P(text=meta_text)
            doc.text.addElement(time_p)
            doc.text.addElement(P(text=""))
            
            # Soru
            q_para = P()
            q_para.addElement(Span(stylename="Question", text="❓ SORU: "))
            q_para.addElement(Span(stylename="Answer", text=qa["question"]))
            doc.text.addElement(q_para)
            doc.text.addElement(P(text=""))
            
            # Kaynaklar
            if qa.get("sources"):
                s_para = P()
                s_para.addElement(Span(stylename="Source", text="📚 KAYNAKLAR:"))
                doc.text.addElement(s_para)
                
                for source in qa["sources"]:
                    source_text = (
                        f"  • {source['name']} "
                        f"(Sayfa {source['page']}) - "
                        f"Benzerlik: {source['score']:.3f}"
                    )
                    source_p = P()
                    source_p.addElement(Span(stylename="Source", text=source_text))
                    doc.text.addElement(source_p)
                
                doc.text.addElement(P(text=""))
            
            # Cevap - Markdown parse et
            c_header = P()
            c_header.addElement(Span(stylename="Question", text="💬 CEVAP:"))
            doc.text.addElement(c_header)
            doc.text.addElement(P(text=""))
            
            # Cevabı satırlara böl ve markdown parse et
            answer_lines = qa["answer"].split('\n')
            
            in_code_block = False
            in_table = False
            table_lines = []
            i = 0
            
            while i < len(answer_lines):
                line = answer_lines[i]
                
                # Kod bloğu kontrolü (```)
                if line.strip().startswith("```"):
                    in_code_block = not in_code_block
                    i += 1
                    continue
                
                if in_code_block:
                    # Kod bloğu içindeyiz - monospace
                    code_p = P()
                    code_p.addElement(Span(stylename="Code", text=line))
                    doc.text.addElement(code_p)
                    i += 1
                    continue
                
                # Tablo başlangıcı kontrolü
                if line.strip().startswith('|') and line.strip().endswith('|'):
                    # Tablo başladı
                    table_lines = [line]
                    # Sonraki satırları kontrol et
                    j = i + 1
                    while j < len(answer_lines):
                        next_line = answer_lines[j].strip()
                        # Tablo satırı mı? (| ile başlayıp | ile biten)
                        if next_line.startswith('|') and next_line.endswith('|'):
                            table_lines.append(next_line)
                            j += 1
                        else:
                            # Boş satır veya separator satırı kontrol et
                            if next_line == '' or re.match(r'^\|[-:\s|]+\|$', next_line):
                                table_lines.append(next_line)
                                j += 1
                            else:
                                break
                    
                    # Tabloyu parse et ve ekle
                    if len(table_lines) >= 2:  # En az başlık ve separator olmalı
                        table, new_idx = self.parse_markdown_table(answer_lines, i)
                        if table:
                            # Tablo başlığı ekle (bir önceki satır tablo başlığı olabilir)
                            if i > 0:
                                prev_line = answer_lines[i-1].strip()
                                # Eğer önceki satır **text** formatında bir başlıksa
                                if prev_line.startswith('**') and prev_line.endswith('**'):
                                    title_text = prev_line[2:-2].strip()
                                    title_para = P(stylename="TableTitle")
                                    title_para.addElement(Span(stylename="Bold", text=title_text))
                                    doc.text.addElement(title_para)
                                elif prev_line and not prev_line.startswith('|') and not prev_line.startswith('```'):
                                    # Normal başlık
                                    title_para = P(stylename="TableTitle")
                                    title_para.addElement(Span(stylename="Bold", text=prev_line))
                                    doc.text.addElement(title_para)
                            
                            # Tabloyu ekle
                            doc.text.addElement(table)
                            doc.text.addElement(P(text=""))  # Boş satır
                            i = new_idx
                            continue
                
                # Normal satır - markdown parse et
                if line.strip():
                    # **text** formatında başlık kontrolü
                    stripped_line = line.strip()
                    if stripped_line.startswith('**') and stripped_line.endswith('**'):
                        # Yıldızlı başlığı Heading2 stilinde göster
                        heading_text = stripped_line[2:-2].strip()
                        heading = H(outlinelevel=2, stylename="Heading2", text=heading_text)
                        doc.text.addElement(heading)
                    else:
                        parsed_para = self.parse_markdown_line(line, doc, {})
                        if parsed_para:
                            doc.text.addElement(parsed_para)
                else:
                    # Boş satır
                    doc.text.addElement(P(text=""))
                
                i += 1
            
            # Ayraç (son öğe değilse)
            if idx < len(qa_pairs):
                doc.text.addElement(P(text=""))
                separator = P(stylename="Separator", text="─" * 60)
                doc.text.addElement(separator)
                doc.text.addElement(P(text=""))
        
        # Dosyayı kaydet
        doc.save(str(filepath))
        
        return filepath
    
    def export_last_qa(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]] = None,
        filename: str = None,
        model_info: Dict[str, Any] = None,
        performance_metrics: Dict[str, Any] = None,
        mode: str = None
    ) -> Path:
        """
        Sadece son soru-cevabı export et.
        
        Args:
            question: Soru metni
            answer: Cevap metni
            sources: Kaynak listesi
            filename: Dosya adı
            model_info: Model bilgileri
            performance_metrics: Performans metrikleri
            mode: Çalışma modu (simple/agentic)
            
        Returns:
            Path: Kaydedilen dosyanın yolu
        """
        # Tarih formatını uygun hale getir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Eğer filename None ise, otomatik oluştur
        if filename is None:
            filename = self._generate_filename(question, answer, timestamp)
        
        qa_pair = {
            "question": question,
            "answer": answer,
            "sources": sources or [],
            "timestamp": datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
            "mode": mode
        }
        
        return self.export_conversation(
            [qa_pair], 
            filename, 
            model_info, 
            performance_metrics
        )
