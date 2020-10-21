from flask import render_template
from app import app

@app.route('/')
def home():
   return "<b>Base</b>"

@app.route('/template')
def template():
    return render_template('home.html')
