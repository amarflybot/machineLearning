from flask import Flask,request

app = Flask(__name__)


@app.route('/',methods=['GET','POST'])
def hello_world():
    if request.method == 'POST':
        if request.form['user']:
            return 'Hello User: '+request.form['user']
    else: return 'Hello Amar from GET!'

if __name__ == '__main__':
    app.run()
