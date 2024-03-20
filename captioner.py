from transformers import pipeline


class Captioner:
    def __init__(self, PROMPT="The main subject of this picture is a"):
        print("Initializing captioner...")

        self.PROMPT = PROMPT

        self.captioner = pipeline(
            "image-to-text",
            model="Salesforce/blip-image-captioning-base",
            prompt=PROMPT
        )

    def derive_caption(self, image, max_new_tokens=20):
        result = self.captioner(image, max_new_tokens)
        raw_caption = result[0]["generated_text"]
        caption = raw_caption.lower().replace(self.PROMPT.lower(), "").strip()
        return caption