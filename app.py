import gradio as gr
from transformers import pipeline
import torch
import html

MODEL_ID = "DScomp380/vit-b16-plant_village"

# The model downloads automatically the first time the Space starts.
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("image-classification", model=MODEL_ID, device=device)

DISEASE_INFO = {
    "strawberry leaf scorch": {
        "cause": "A fungal disease commonly associated with Diplocarpon earlianum.",
        "symptoms": "Small purple, red-brown or dark spots; affected areas may merge and leaves can look scorched.",
        "care": "Remove fallen infected leaves, improve airflow, avoid prolonged leaf wetness and seek local agricultural advice if the problem spreads."
    },
    "tomato early blight": {
        "cause": "A fungal disease commonly associated with Alternaria species.",
        "symptoms": "Brown lesions, often with concentric rings, frequently beginning on older leaves.",
        "care": "Remove infected debris, mulch soil to reduce splash, improve airflow and avoid overhead watering."
    },
    "tomato late blight": {
        "cause": "A disease caused by the oomycete Phytophthora infestans.",
        "symptoms": "Dark, water-soaked lesions that may spread quickly in cool, humid conditions.",
        "care": "Avoid prolonged leaf wetness, remove affected material safely and consult local agricultural guidance."
    },
    "potato early blight": {
        "cause": "A fungal disease commonly associated with Alternaria species.",
        "symptoms": "Brown leaf lesions that can develop target-like rings.",
        "care": "Use healthy planting material, rotate crops, remove infected debris and avoid wetting foliage when watering."
    },
    "potato late blight": {
        "cause": "A disease caused by the oomycete Phytophthora infestans.",
        "symptoms": "Dark, water-soaked patches on leaves; pale growth may appear around lesions in humid conditions.",
        "care": "Reduce leaf wetness, remove affected material safely and contact a local agricultural expert if suspected."
    },
    "pepper bell bacterial spot": {
        "cause": "A bacterial disease associated with Xanthomonas species.",
        "symptoms": "Small water-soaked spots that can become dark or necrotic.",
        "care": "Use clean seeds or transplants, avoid handling wet plants, sanitize tools and seek local agricultural guidance."
    },
    "tomato bacterial spot": {
        "cause": "A bacterial disease associated with Xanthomonas species.",
        "symptoms": "Small dark spots on leaves and sometimes lesions on stems or fruit.",
        "care": "Use clean planting material, avoid working among wet plants, sanitize tools and remove infected debris."
    },
    "tomato leaf mold": {
        "cause": "A fungal disease commonly associated with Passalora fulva.",
        "symptoms": "Pale yellow patches on upper leaf surfaces and olive or greyish growth underneath in humid conditions.",
        "care": "Improve ventilation, reduce humidity around foliage and remove affected leaves carefully."
    },
    "tomato septoria leaf spot": {
        "cause": "A fungal disease caused by Septoria lycopersici.",
        "symptoms": "Small circular spots with pale centres and dark margins, often on lower leaves.",
        "care": "Remove infected lower leaves and plant debris, mulch soil and avoid overhead watering."
    },
    "tomato yellow leaf curl virus": {
        "cause": "A viral disease transmitted by whiteflies.",
        "symptoms": "Yellowing, upward curling leaves and stunted growth may occur.",
        "care": "Inspect for whiteflies and seek local agricultural advice about management."
    },
    "healthy": {
        "cause": "The model classified the image as a healthy category.",
        "symptoms": "The model did not select a disease pattern as its top prediction.",
        "care": "Continue suitable watering, lighting and nutrition. Inspect plants regularly; a model prediction cannot guarantee that a plant is disease-free."
    }
}

def clean_label(label):
    return " ".join(label.replace("___", " - ").replace("_", " ").replace(",", "").split())

def find_info(label):
    normalized = clean_label(label).lower()
    for key, info in DISEASE_INFO.items():
        if key in normalized:
            return info
    if "healthy" in normalized:
        return DISEASE_INFO["healthy"]
    return None

def analyze_leaf(image):
    if image is None:
        return {}, "Please upload a clear photograph of a plant leaf."

    predictions = classifier(image.convert("RGB"), top_k=3)
    top = predictions[0]
    label = top["label"]
    score = float(top["score"])
    display_label = clean_label(label)

    scores = {clean_label(p["label"]): float(p["score"]) for p in predictions}
    normalized = display_label.lower()

    if "background without leaves" in normalized:
        status = "⚠️ No leaf detected"
        note = "The model thinks this image may not contain a clear leaf. Try a close-up photograph."
    elif "healthy" in normalized:
        status = "🌿 Predicted healthy"
        note = "This is a model prediction, not proof that the plant is disease-free."
    else:
        status = "🔎 Possible disease pattern detected"
        note = "The result is a preliminary AI prediction, not a confirmed diagnosis."

    info = find_info(label)
    if info:
        details = f"""### Possible cause
{info['cause']}

### Common symptoms
{info['symptoms']}

### General care and prevention
{info['care']}"""
    else:
        details = """### General plant-care guidance
- Examine both sides of the leaf and check other leaves on the same plant.
- Take a clear photo in natural light.
- Avoid unnecessary leaf wetness and remove fallen plant debris when appropriate.
- Consult a local agricultural expert before choosing a treatment."""

    confidence_note = (
        " **Low confidence:** treat this result as uncertain and try a clearer image."
        if score < 0.60 else
        " A higher score does not guarantee that the prediction is correct."
    )
    report = f"""## {status}

**Top prediction:** {html.escape(display_label)}  
**Model confidence:** {score * 100:.1f}%

{note}{confidence_note}

---

{details}

---

> **Important:** This tool is for an educational project and preliminary screening only. Do not apply pesticides or other treatments solely based on its output.
"""
    return scores, report

CSS = """
.gradio-container {max-width: 1120px !important; margin: auto !important;}
#hero {text-align: center; padding: 12px 0 4px 0;}
"""

with gr.Blocks(
    title="AI Plant Disease Detection & Health Advisor",
    theme=gr.themes.Soft(primary_hue="green", secondary_hue="lime"),
    css=CSS
) as demo:
    gr.Markdown(
        """# 🌱 AI Plant Disease Detection
### Plant Health Advisory System

Upload a clear leaf image to see the model's top predictions,
confidence scores and general educational plant-care guidance.""",
        elem_id="hero"
    )
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil", label="Upload a plant leaf image",
                sources=["upload", "webcam"]
            )
            with gr.Row():
                analyze_button = gr.Button("🔍 Analyze leaf", variant="primary")
                clear_button = gr.ClearButton(value="Clear", components=[image_input])
            gr.Markdown("**Best results:** use a sharp, well-lit close-up of one leaf. Avoid blurry images and busy backgrounds.")
        with gr.Column(scale=1):
            prediction_output = gr.Label(label="Top AI predictions", num_top_classes=3)
            report_output = gr.Markdown("Your plant health report will appear here after analysis.")

    analyze_button.click(
        fn=analyze_leaf,
        inputs=image_input,
        outputs=[prediction_output, report_output]
    )
    gr.Markdown("""---
**Built with:** Python · Gradio · Hugging Face Transformers · Vision Transformer (ViT)

**Educational prototype:** predictions may be wrong, especially for real-world images that differ from the model's training dataset. Verify disease concerns with a qualified agricultural professional.""")

if __name__ == "__main__":
    demo.launch()
