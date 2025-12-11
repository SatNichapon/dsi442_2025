import gradio as gr
import torch
import numpy as np
import os
import librosa
import plotly.graph_objects as go 
from src.features import FeatureEngine
from src.model import DigitalSoulModel
import config

# --- 1. SETUP ---
device = config.DEVICE
print(f"🚀 Launching App on {device}...")

# Load Model
model = DigitalSoulModel().to(device)
checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "digital_soul_final.pth")

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("✅ Model weights loaded!")
else:
    print("⚠️ No trained model found! Using random weights.")

model.eval()

# Initialize Feature Engine
print("⏳ Initializing Feature Engine...")
engine = FeatureEngine()
print("✅ Feature Engine Ready.")

# --- 2. VISUALIZATION (Neon Cyberpunk Style) ---
def generate_interactive_radar(scores):
    labels = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
    plot_scores = np.concatenate((scores, [scores[0]]))
    plot_labels = labels + [labels[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=plot_scores,
        theta=plot_labels,
        fill='toself',
        name='Personality',
        line=dict(color='#00f2ea', width=3), 
        fillcolor='rgba(138, 43, 226, 0.3)', 
        marker=dict(size=6, color='#fff')
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        polar=dict(
            bgcolor="rgba(20, 20, 30, 0.5)", 
            radialaxis=dict(
                visible=True, 
                range=[0, 1], 
                showticklabels=False, 
                gridcolor='rgba(255, 255, 255, 0.1)'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, family="Roboto", color="#00f2ea"),
                rotation=90,
                direction="clockwise",
                gridcolor='rgba(255, 255, 255, 0.1)'
            )
        ),
        showlegend=False,
        dragmode=False, 
        margin=dict(l=40, r=40, t=40, b=40),
        height=400,
        title={'text': ""} 
    )
    
    return fig

def predict_personality(audio_file):
    print(f"\n🖱️ Processing: {audio_file}")

    if audio_file is None:
        return None, "Please upload an audio file."

    # Prevents crash if browser sends empty file (common with quick mic toggling)
    if not os.path.exists(audio_file) or os.path.getsize(audio_file) == 0:
        return None, "⚠️ Audio file is empty. Please wait a moment after recording before clicking."

    try:
        y, sr = librosa.load(audio_file, sr=config.SAMPLE_Rate, duration=config.MAX_DURATION)
        target_len = config.SAMPLE_Rate * config.MAX_DURATION
        if len(y) < target_len:
            y = librosa.util.fix_length(y, size=target_len)
        else:
            y = y[:target_len]
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        res = engine.whisper.transcribe(audio_file)
        text = res['text'].strip() or "silence"
        
        inputs = engine.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=config.MAX_TEXT_LEN).to(device)
        with torch.no_grad():
            outputs = engine.text_model(**inputs)
        ling_vec = outputs.last_hidden_state[:, 0, :].cpu().numpy().squeeze()

        acou_t = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device).transpose(1, 2)
        ling_t = torch.tensor(ling_vec, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            prediction = model(ling_t, acou_t).cpu().numpy()[0]

        chart = generate_interactive_radar(prediction)
        
        traits = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism']
        word_count = len(text.split())
        
        # HTML Content
        html_content = f"""
        <div style="background-color: rgba(255,255,255,0.05); padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h3 style="color: #00f2ea; margin-top: 0;">📝 Transcription</h3>
            <p style="font-style: italic; color: #eee;">"{text}"</p>
            <small style="color: #ccc;">Analyzed {word_count} words</small>
        </div>
        <div style="margin-top: 10px;">
        """
        
        for t, s in zip(traits, prediction):
            pct = int(s * 100)
            color = "#00f2ea" if s > 0.6 else "#ff0055" if s < 0.4 else "#f2e205"
            html_content += f"""
            <div style="margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; color: white; font-weight: bold;">
                    <span>{t}</span>
                    <span style="color: {color};">{s:.2f}</span>
                </div>
                <div style="background-color: #333; border-radius: 5px; height: 8px; width: 100%;">
                    <div style="background-color: {color}; width: {pct}%; height: 100%; border-radius: 5px; box-shadow: 0 0 5px {color};"></div>
                </div>
            </div>
            """
        html_content += "</div>"

        return chart, html_content

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error processing audio: {str(e)}"

# --- 3. UI SETUP ---

custom_css = """
/* Main Background */
body, .gradio-container {
    background-color: #0b0f19 !important; 
    color: white !important;
}
/* Panels & Blocks */
.block, .panel, .pad {
    background-color: #111625 !important;
    border: 1px solid #333 !important;
}
/* Typography */
h1 {
    color: #00f2ea !important; 
    text-shadow: 0 0 10px rgba(0, 242, 234, 0.5); 
    text-align: center;
}
/* Force all headers (h2, h3, h4) to be visible */
h2, h3, h4 {
    color: #e0e0e0 !important;
}
/* Specifically target Markdown strong/bold text */
strong {
    color: #00f2ea !important;
}
/* Standard text paragraphs */
p, span, li, .prose {
    color: #cccccc !important;
}
/* Accordion labels */
span.label-wrap {
    color: #ffffff !important;
}
/* Subtitle specific class */
.subtitle {
    color: #aaaaaa !important; 
    text-align: center; 
    margin-bottom: 20px;
    font-size: 1.1em;
}
/* Input Fields */
input, textarea, .gr-input {
    background-color: #050810 !important;
    color: white !important;
    border: 1px solid #333 !important;
}
/* Buttons */
button.primary {
    background: linear-gradient(90deg, #00f2ea 0%, #00cc96 100%) !important;
    color: black !important;
    font-weight: bold !important;
    border: none !important;
}
"""

with gr.Blocks(title="Digital Soul") as demo:
    
    gr.HTML(f"<style>{custom_css}</style>")
    
    gr.Markdown("# 🧠 DIGITAL SOUL")
    gr.Markdown("<div class='subtitle'>Multi-Modal Personality Analysis Engine</div>")
    
    with gr.Row():
        # LEFT COLUMN (Input)
        with gr.Column(scale=1):
            gr.Markdown("### 🎤 Input Signal")
            audio_input = gr.Audio(type="filepath", sources=["upload", "microphone"], label="Audio Source")
            submit_btn = gr.Button("INITIALIZE SCAN", variant="primary", size="lg")
            
            with gr.Accordion("ℹ️ System Architecture", open=False):
                gr.Markdown(
                    """
                    **Linguistic Stream:** ModernBERT (8k Context) \n
                    **Acoustic Stream:** Deep Learning (LSTM) \n
                    **Fusion:** Late-Stage Concatenation
                    """
                )

        # RIGHT COLUMN (Dashboard)
        with gr.Column(scale=2):
            with gr.Row():
                # Chart
                with gr.Column(scale=1):
                    gr.Markdown("### 🔮 Fingerprint")
                    output_plot = gr.Plot(label="Radar", container=False)
                
                # Stats
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Metrics")
                    output_html = gr.HTML(label="Analysis")

            # --- Trait Guide ---
            with gr.Accordion("📘 Trait Definitions (What do these scores mean?)", open=False):
                gr.Markdown(
                    """
                    * **Openness:** Imagination, insight, and eagerness to learn. High scores indicate curiosity and creativity.
                    * **Conscientiousness:** Reliability, organization, and discipline. High scores indicate a goal-oriented mindset.
                    * **Extraversion:** Energy, sociability, and assertiveness. High scores indicate an outgoing and talkative nature.
                    * **Agreeableness:** Kindness, trust, and cooperation. High scores indicate a helpful and empathetic personality.
                    * **Neuroticism:** Sensitivity to stress and negative emotions. High scores indicate frequent worry or mood fluctuations.
                    """
                )

    submit_btn.click(
        fn=predict_personality,
        inputs=audio_input,
        outputs=[output_plot, output_html]
    )

if __name__ == "__main__":
    demo.launch()