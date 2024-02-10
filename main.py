from flask import Flask, render_template, request, redirect, url_for
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    topic = request.form['topic']
    return redirect(url_for('result', topic=topic))
    
@app.route('/result/<topic>')
def result(topic):
    return render_template('result.html', topic=topic)

@app.route('/video/<topic>/generate', methods=['POST'])
def generate(topic):
    if request.method == 'POST':
        topic = sanitize_input(request.form['topic'])
        print(topic)

        # Generate script
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("gpt2")

        inputs = tokenizer.encode(topic, return_tensors='pt')
        outputs = model.generate(inputs, max_length=500, temperature=0.7, num_return_sequences=1)

        script = tokenizer.decode(outputs[0])
        print(script)  # Print the generated script

        return redirect(url_for('video', topic=topic, script=script))

@app.route('/video/<topic>')
def video(topic):
    script = request.args.get('script')
    return render_template('video.html', topic=topic,script=script)

@app.route('/generate_video', methods=['POST'])
def generate_video():
    topic = sanitize_input(request.form.get('topic'))
    
    if not topic:
        return render_template('error.html', message='Invalid input')
    
    # Now you can use the 'topic' variable to generate a video
    # ...

    return render_template('result.html', topic=topic)

def sanitize_input(input_string):
    # Remove any non-alphanumeric characters from the input string
    sanitized_string = re.sub(r'\W+', '', input_string)
    return sanitized_string

if __name__ == '__main__':
    app.run(debug=True)
