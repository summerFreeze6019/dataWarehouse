import pickle, yaml
from lib.learning import predict
from lib.camera import capture_image

from flask import Flask, jsonify, request, abort
app = Flask(__name__)

with open('classifier.pickle') as f:
    clf = pickle.load(f)
with(open('settings.yml')) as f:
    settings = yaml.load(f)


class ValidationError(Exception):
    pass


def validate_user(username, password):
    if settings['users'][username] != password:
        raise ValidationError()


@app.route('/<int:token_id>', methods=['POST'])
def index(token_id):
    try:
        username = request.values['username']
        password = request.values['password']
        validate_user(username, password)
        boundaries = settings['crop'][token_id]
    except KeyError, ValidationError:
        abort(401)
    
    img = capture_image()
    prediction, binarized_img, chars = predict(clf, img=img,
        boundaries=boundaries)
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)