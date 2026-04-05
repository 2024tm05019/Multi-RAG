"""
Vision Model: Summarizes images using GPT-4o.
Converts circuit diagrams, connector pinouts, and wiring images into text.
"""

import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def summarize_image(image_b64: str, ext: str, page: int) -> str:
    """
    Send image to GPT-4o and get a text description.
    Tailored prompt for industrial/electrical technical manuals.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"This image is from page {page} of a Bosch Rexroth weld timer "
                                "technical manual. Describe in detail what this image shows. "
                                "If it is a circuit diagram, describe the components and connections. "
                                "If it is a table or pinout diagram, list all labels, pin numbers, "
                                "and values. If it is a mechanical drawing, describe dimensions and parts. "
                                "Be specific and technical."
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{ext};base64,{image_b64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Image on page {page}: could not be summarized. Error: {str(e)}"
