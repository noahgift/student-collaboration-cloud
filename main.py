import logging

from flask import Flask, render_template
from multiprocessing import Process
import os
import sys

import sokoban
app = Flask(__name__)

def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())

def f():
    ## This could accept additiona commands vs f(algorithm), etc
    sys.stdout = open(str(os.getpid()) + ".out", "w")
    ## You could write this out to Google Cloud Storage 
    ## Be aware that Google App Engine file system is read only
    info('function f')
    sokoban.main()

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'

@app.route("/sokoban")
def easy_sokoban():
    p = Process(target=f)
    p.start()
    p.join()
    ### Read from Google Cloud Storage?
    return "Stuff"


@app.route('/newroute/<name>')
def newroute(name):
    """parameter"""
    return "this was passed in: %s" % name


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500


if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    app.run(host='127.0.0.1', port=8080, use_reloader=True, debug=True)
