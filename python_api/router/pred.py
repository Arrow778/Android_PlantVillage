from flask import Blueprint ,render_template,abort,jsonify,request
pred_bp = Blueprint('pred', __name__)


@pred_bp.route("/pred",methods=['POST'])
def predict():
    """
    predict 的 Docstring
    这里应该是一个预测的逻辑，
    Android端调用这个接口，需要传入图片bit码，返回预测结果
    采用RESTful 接口
    """ 
    if request.method != "POST":
        return jsonify({"error": "method not allowed"}), 405
    return jsonify({"result": "success","predict_id":5}), 200
    