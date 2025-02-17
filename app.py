from flask import Flask, request, jsonify, render_template, Response
from env.env import get_api_key
from script.vectorstore import loadFaiss
from BllossomModule import MakeTable, Conversation

app = Flask(__name__)

# API KEY 정보를 가져온다.
get_api_key()
# FAISS 데이터베이스를 로드한다.
loadFaiss()


@app.route("/")
def home():
    # return to page/index.html
    return render_template("index.html")


@app.route("/api", methods=["POST"])
def api():
    data = request.json
    name = data.get("name")
    response = {"message": f"Hello, {name}!"}
    return jsonify(response)


@app.route("/api/questpost", methods=["POST"])
def api_questpost():
    # #data = request.json
    text = request.form.get('textInput')
    data = request.files.get('pdfFile')

    if data:
        data = data.read()
        return MakeTable(data) 
    elif text:
        return Conversation(text)

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5147)
