import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Patch

def create_figure_i2_visualization():
    """
    DO-260C Figure I-2 - Current and Enhanced Bit Demodulation Techniques
    Görselleştirme fonksiyonu
    """
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
    fig.suptitle('Figure I-2: Current and Enhanced Bit Demodulation Techniques', 
                 fontsize=14, fontweight='bold')
    
    # Genel parametreler
    chip_width = 0.3
    pulse_height = 1.0
    
    # Part A - Basit Merkez Örnekleme Tekniği (Simple Center Sample Technique)
    ax_a = axes[0]
    x = np.linspace(0, 2, 400)
    
    # Preamble pulse reference
    preamble_x = np.array([0.5, 0.6])
    preamble_y = [pulse_height * 0.8, pulse_height * 0.8]
    ax_a.plot(preamble_x, preamble_y, 'k-', linewidth=2, label='Preamble Reference')
    
    # Data bits with center sampling
    for i in range(4):
        chip_center = i + 1
        bit_value = np.random.choice([0.5, 0.8])  # Simulated amplitude
        
        ax_a.plot([chip_center - chip_width/2, chip_center - chip_width/2], 
                  [0, pulse_height * 0.3], 'b-', linewidth=1)
        ax_a.plot([chip_center + chip_width/2, chip_center + chip_width/2], 
                  [0, pulse_height * bit_value], 'r-', linewidth=1)
        
        # Center sample point
        center_amp = (bit_value + 0.5) / 2
        ax_a.plot(chip_center, center_amp, 'go', markersize=8, label='Center Sample' if i==0 else "")
    
    ax_a.set_xlim(0, 5)
    ax_a.set_ylim(0, pulse_height * 1.2)
    ax_a.set_xlabel('Time (Chip Units)', fontsize=10)
    ax_a.set_ylabel('Amplitude', fontsize=10)
    ax_a.set_title('(a) Center Sample Technique', fontweight='bold')
    ax_a.grid(True, alpha=0.3)
    
    # Part B - Tüm Örneklem Kullanımı (All Samples Per Chip)
    ax_b = axes[1]
    
    for i in range(4):
        chip_center = i + 1
        
        # Multiple samples per chip
        sample_points = np.linspace(chip_center - chip_width/2, 
                                   chip_center + chip_width/2, 5)
        
        for j, sx in enumerate(sample_points):
            amp = pulse_height * (0.6 if j < 3 else 0.9)
            ax_b.plot(sx, amp, 'bo', markersize=4)
            
            # Bit decision line
            if j == len(sample_points) - 1:
                bit_level = np.mean([pulse_height * 0.6, pulse_height * 0.9])
                ax_b.axhline(y=bit_level, color='r', linestyle='--', alpha=0.5)
    
    ax_b.set_xlim(0, 5)
    ax_b.set_ylim(0, pulse_height * 1.2)
    ax_b.set_title('(b) All Samples Per Chip Technique', fontweight='bold')
    ax_b.grid(True, alpha=0.3)
    
    # Part C - Zayıf Sinyal Güçlü Sinyal Karışımı (Signal Overlap Scenario)
    ax_c = axes[2]
    
    # Strong signal (later in time)
    strong_x = np.linspace(1.5, 4.0, 200)
    strong_y = pulse_height * 0.9
    ax_c.plot(strong_x, strong_y, 'r-', linewidth=2, label='Strong Signal')
    
    # Weak signal (earlier in time) - overlapped
    weak_x = np.linspace(0.5, 3.0, 200)
    weak_y = pulse_height * 0.4
    ax_c.plot(weak_x, weak_y, 'b-', linewidth=1.5, label='Weak Signal')
    
    # Overlap region showing bit errors
    overlap_region = Rectangle((1.5, 0), 1.5, pulse_height * 0.9, 
                               fill=False, edgecolor='orange', linewidth=2, linestyle='--')
    ax_c.add_patch(overlap_region)
    
    # Bit error indication
    for i in range(3):
        error_x = 1.8 + i * 0.5
        ax_c.plot([error_x, error_x], [pulse_height*0.4, pulse_height*0.9], 
                  'k-', linewidth=2)
    
    ax_c.set_xlim(0, 5)
    ax_c.set_ylim(0, pulse_height * 1.2)
    ax_c.set_title('(c) Signal Overlap (Bit Errors)', fontweight='bold')
    ax_c.grid(True, alpha=0.3)
    
    # Part D - Amplitude Correlation ile İyileştirilmiş Teknik
    ax_d = axes[3]
    
    for i in range(4):
        chip_center = i + 1
        
        # Preamble correlation reference
        preamble_amp = pulse_height * 0.85
        
        # Data with amplitude correlation
        data_amp = pulse_height * (0.7 if i % 2 == 0 else 0.9)
        
        ax_d.plot(chip_center, preamble_amp, 'ko', markersize=6, 
                  label='Preamble Correlation' if i==0 else "")
        ax_d.plot(chip_center, data_amp, 'ro', markersize=8)
        
        # Confidence level indicator
        confidence = 0.9 if abs(data_amp - preamble_amp) < 0.15 else 0.6
        ax_d.text(chip_center, pulse_height * 0.3, f'Conf: {confidence:.2f}', 
                 ha='center', fontsize=8)
    
    ax_d.set_xlim(0, 5)
    ax_d.set_ylim(0, pulse_height * 1.2)
    ax_d.set_title('(d) Amplitude Correlation Technique', fontweight='bold')
    ax_d.grid(True, alpha=0.3)
    
    # Add legend to first subplot only
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.02), 
              ncol=4, fontsize=10)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig

def create_detailed_bit_demodulation_chart():
    """
    Daha detaylı bit demodülasyon teknikleri karşılaştırma grafiği
    """
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Teknik isimleri ve özellikleri
    techniques = [
        'Merkez Örneklem',
        'Tüm Örneklem Kullanımı', 
        'Zayıf Sinyal Karışımı',
        'Amplitude Korelasyon'
    ]
    
    accuracy_scores = [0.85, 0.92, 0.73, 0.96]
    complexity_scores = [1.0, 2.5, 1.5, 2.0]
    error_rates = [0.15, 0.08, 0.27, 0.04]
    
    x = np.arange(len(techniques))
    width = 0.25
    
    bars1 = ax.bar(x - width, accuracy_scores, width, label='Doğruluk (%)', color='green')
    bars2 = ax.bar(x, complexity_scores, width, label='Karmaşıklık', color='blue')
    bars3 = ax.bar(x + width, error_rates, width, label='Hata Oranı', color='red')
    
    ax.set_ylabel('Skor Değerleri', fontsize=12)
    ax.set_title('Bit Demodülasyon Tekniklerinin Karşılaştırılması', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(techniques, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Değerleri ekle
    for i, v in enumerate(accuracy_scores):
        ax.text(i - width, v + 0.02, f'{v:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig

def create_confidence_declaration_chart():
    """
    Güvenlik Beyanı (Confidence Declaration) Görselleştirme
    """
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    # Üst grafik - Bit Karar Süreci
    ax1 = axes[0]
    chip_times = np.linspace(0, 4, 100)
    
    for i in range(5):
        start = i * 0.8
        end = min(start + 0.6, 3.9)
        
        if i % 2 == 0:
            amplitude = np.random.uniform(0.7, 0.9)
            bit_value = 1
        else:
            amplitude = np.random.uniform(0.4, 0.6)
            bit_value = 0
        
        ax1.plot([start, end], [amplitude * pulse_height] * 2, 
                'k-', linewidth=2 if i==2 else 1)
        
        # Bit decision threshold
        if i == 2:
            ax1.axhline(y=pulse_height * 0.65, color='r', linestyle='--', alpha=0.7)
            ax1.text(3.5, pulse_height * 0.65, 'Threshold', fontsize=8)
    
    ax1.set_ylim(0, pulse_height * 1.2)
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_title('(a) Bit Decision Process with Confidence Levels', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Alt grafik - Güvenlik Seviyeleri
    ax2 = axes[1]
    confidence_levels = ['Düşük', 'Orta', 'Yüksek']
    confidence_values = [0.5, 0.75, 0.95]
    
    colors = ['#ffcccc', '#ccffcc', '#ccffff']
    
    for i, (level, value) in enumerate(zip(confidence_levels, confidence_values)):
        ax2.barh(i, value, color=colors[i], alpha=0.8)
        ax2.text(value + 0.05, i, f'{value:.2f}', va='center', fontsize=10)
    
    ax2.set_xlim(0, 1.1)
    ax2.set_yticks(range(len(confidence_levels)))
    ax2.set_yticklabels(confidence_levels)
    ax2.set_xlabel('Confidence Level (0-1)', fontsize=10)
    ax2.set_title('(b) Confidence Declaration Levels', fontweight='bold')
    
    plt.tight_layout()
    return fig

# Kodu çalıştır ve görselleri kaydet
if __name__ == "__main__":
    print("Figure I-2 Görselleştirme Kodu Çalıştırılıyor...")
    
    # Ana görsel
    fig1 = create_figure_i2_visualization()
    plt.savefig('figure_i2_bit_demodulation.png', dpi=300, bbox_inches='tight')
    print("✓ Figure I-2 ana görseli kaydedildi: figure_i2_bit_demodulation.png")
    
    # Detaylı karşılaştırma grafiği
    fig2 = create_detailed_bit_demodulation_chart()
    plt.savefig('figure_i2_technique_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Teknik karşılaştırma grafiği kaydedildi: figure_i2_technique_comparison.png")
    
    # Güvenlik beyanı görselleştirmesi
    fig3 = create_confidence_declaration_chart()
    plt.savefig('figure_i2_confidence_levels.png', dpi=300, bbox_inches='tight')
    print("✓ Güvenlik seviyeleri grafiği kaydedildi: figure_i2_confidence_levels.png")
    
    plt.show()
