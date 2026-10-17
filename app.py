from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# --- تدريب النموذج عند تشغيل السيرفر ---
print("جاري تدريب النموذج في الخلفية، يرجى الانتظار ثوانٍ...")
df = pd.read_csv('enron.csv', usecols=['Message'], nrows=20000).dropna()
spam_keywords = ['money', 'free', 'offer', 'win', 'urgent', 'bank', 'lottery']
pattern = '|'.join(spam_keywords)
df['label'] = np.where(df['Message'].str.contains(pattern, case=False, na=False), 1, 0)

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_weights = vectorizer.fit_transform(df['Message'])
model = MultinomialNB()
model.fit(X_weights, df['label'])
print("تم التدريب بنجاح! السيرفر جاهز الآن.")

# --- مسارات تطبيق الويب ---
@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        # استقبال النص من الواجهة
        user_message = request.form['message']
        # تحليل النص
        input_weights = vectorizer.transform([user_message])
        prediction = model.predict(input_weights)[0]
        # تحديد النتيجة
        result = 'مشبوهة' if prediction == 1 else 'سليمة'
        
    return render_template('index.html', result=result)

if __name__ == '__main__':
    # تشغيل السيرفر
    app.run(debug=True)