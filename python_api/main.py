from flask import Flask
from router.pred import pred_bp

app = Flask(__name__)
app.register_blueprint(pred_bp,url_prefix="/pred")
@app.route("/")
@app.route("/app")
def main():
    return "Hello World!"

if __name__ == "__main__":
    app.run(debug=True,port=5002)