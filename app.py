import os
from flask import Flask, request, jsonify
from openai import OpenAI

app = Flask(__name__)

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@app.get("/")
def health():
    return jsonify({
        "status": "ok",
        "service": "alice-el-backend"
    })


@app.post("/ask")
def ask():
    try:
        data = request.get_json(silent=True) or {}
        text = data.get("text", "").strip()

        if not text:
            return jsonify({"error": "No text provided"}), 400

        response = client.responses.create(
            model="gpt-5-mini",
            input=text
        )

        return jsonify({
            "answer": response.output_text
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
