# Import necessary packages
from flask import Flask, render_template, request
import io
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader
from docx import Document

from groq_client import generate_completion
from orchestrator import run_pipeline, get_context
from agents.question_generator import generate_questions
from agents.candidate_communicator import draft_email

nltk.download('punkt')
nltk.download('punkt_tab')

app = Flask(__name__)

ALLOWED_UPLOAD_EXTENSIONS = ("pdf", "docx", "txt")


def extract_text_from_upload(file_storage):
    """Extract plain text from an uploaded PDF, DOCX or TXT file."""
    filename = file_storage.filename or ""
    extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if extension not in ALLOWED_UPLOAD_EXTENSIONS:
        raise ValueError(f"Unsupported file type '.{extension}'. Please upload a PDF, DOCX or TXT file.")

    data = file_storage.read()
    if not data:
        raise ValueError(f"'{filename}' is empty.")

    if extension == "pdf":
        reader = PdfReader(io.BytesIO(data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    elif extension == "docx":
        doc = Document(io.BytesIO(data))
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    else:
        text = data.decode("utf-8", errors="ignore")

    text = text.strip()
    if not text:
        raise ValueError(f"Could not extract any text from '{filename}'.")
    return text


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/jd")
def jd():
    return render_template("jd.html")

@app.route("/jdresume")
def jdresume():
    return render_template("jdresume.html")

@app.route("/iqs")
def iqs():
    return render_template("iqs.html")

@app.route("/email")
def email():
    return render_template("email.html")

@app.route("/jd", methods=["POST"])
def jdform():
    try:
        jobposition = str(request.form.get("jobposition"))
        tone = str(request.form.get("tone"))
        skills = str(request.form.get("skills"))
        qualifications = str(request.form.get("qualifications"))
        if jobposition != "" and tone != "" and skills != "" and qualifications != "":
            prompt = "Can you please provide me with the job description for a "+jobposition+"  role in a "+tone+" tone? The ideal candidate should have experience with "+skills+", "+qualifications+". Additionally, I'd like to know more about the company culture and any opportunities for growth within the organization. The output should be in a formatted way on a webpage that use HTML tags to structure the content.Thank you!"
            answer = generate_completion([
                {"role": "system", "content": "You are an assistant who provides job descriptions."},
                {"role": "user", "content": prompt}
            ])
            return render_template("jd.html", result = answer)
        else:
            response = "All fields are mandatory. Please try again."
            return render_template("jd.html", result = response)
    except Exception as e:
        response = "Something went wrong. Please try again.<br><br>"+"Error is: "+str(e)
        return render_template("jd.html", result = response)


@app.route("/iqs", methods=["POST"])
def iqsform():
    try:
        jobposition = str(request.form.get("jobposition"))
        tone = str(request.form.get("tone"))
        skills = str(request.form.get("skills"))
        experience = str(request.form.get("experience"))
        if jobposition != "" and tone != "" and skills != "" and experience != "":
            prompt = "Can you generate the top 10 to 15 interview questions that a hiring manager might ask a candidate for "+jobposition+" with "+experience+" years of experience? The questions should be tailored to the required skills such as "+skills+" for the job and should be asked in a "+tone+" that is appropriate for the company culture. The output should be in a formatted way on a webpage that use HTML tags to structure the content.Thank you!"
            answer = generate_completion([
                {"role": "system", "content": "You are an assistant who creates Interview questions."},
                {"role": "user", "content": prompt}
            ])
            return render_template("iqs.html", result = answer)
        else:
            response = "All fields are mandatory. Please try again."
            return render_template("iqs.html", result = response)
    except Exception as e:
        response = "Something went wrong. Please try again.<br><br>"+"Error is: "+str(e)
        return render_template("iqs.html", result = response)


@app.route("/email", methods=["POST"])
def emailform():
    try:
        emailtype = str(request.form.get("emailtype"))
        tone = str(request.form.get("tone"))
        opcom = str(request.form.get("opcom"))
        if emailtype != "" and tone != "":
            prompt = "Hello, I am looking to generate a customized email template for various job-related purposes. Can you please help me by providing an email template based on the following inputs: Email type as "+emailtype+" and Tone as "+tone+" and additional information such as "+opcom+" and other relevant information. The output should be in a formatted way on a webpage that use HTML tags to structure the content. Thank you!"
            answer = generate_completion([
                {"role": "system", "content": "You are an assistant who can create beautiful and professional emails."},
                {"role": "user", "content": prompt}
            ])
            return render_template("email.html", result = answer)
        else:
            response = "Mandatory fields missing. Please try again."
            return render_template("email.html", result = response)
    except Exception as e:
        response = "Something went wrong. Please try again.<br><br>"+"Error is: "+str(e)
        return render_template("email.html", result = response)


@app.route("/jdresume", methods=["POST"])
def jdresumeform():
    try:
        jd_file = request.files.get("jd_file")
        resume_file = request.files.get("resume_file")
        if jd_file and jd_file.filename and resume_file and resume_file.filename:
            jdtext = extract_text_from_upload(jd_file)
            resumetext = extract_text_from_upload(resume_file)

            nltk.download('stopwords')
            nltk.download('punkt')
            stop_words = set(stopwords.words('english'))
            # Tokenize and clean the text data
            jd_tokens = nltk.word_tokenize(jdtext)
            jd_tokens = [word.lower() for word in jd_tokens if word.isalpha() and word.lower() not in stop_words]
            resume_tokens = nltk.word_tokenize(resumetext)
            resume_tokens = [word.lower() for word in resume_tokens if word.isalpha() and word.lower() not in stop_words]
            # Use TF-IDF to extract important keywords and phrases
            tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
            jd_tfidf = tfidf.fit_transform([jd_tokens])
            resume_tfidf = tfidf.transform([resume_tokens])
            # Calculate similarity score
            similarity_score = cosine_similarity(jd_tfidf, resume_tfidf)[0][0]

            # Shorten the Job Description
            jdprompt = "Please shorten the Job Description to less than 500 words. Here is the JD: " + jdtext
            jdshort = generate_completion([
                {"role": "system", "content": "You are an assistant who summarize job descriptions."},
                {"role": "user", "content": jdprompt}
            ])

            # Shorten the Resume
            resumeprompt = "Please shorten the Resume to less than 500 words. Here is the Resume: " + resumetext
            resumeshort = generate_completion([
                {"role": "system", "content": "You are an assistant who summarize Resume"},
                {"role": "user", "content": resumeprompt}
            ])

            # JD and Resume Comparision
            prompt = "Hi, Here is Job Description: "+jdshort+" and Job Seeker Resume: "+resumeshort+". Check for similarities and differences between the JD and Resume and briefly explain if there is a good match between these to the hiring manager in their own language. Please ensure the completion length is limited to 3000 tokens. The output should be in a formatted way on a webpage that use HTML tags to structure the content. Thank you!"
            answer = generate_completion([
                {"role": "system", "content": "You are an assistant who compares Resume and job descriptions."},
                {"role": "user", "content": prompt}
            ])
            return render_template("jdresume.html", result = "Match Index: <b>" + str("{:.2f}".format(similarity_score*100)) + " %</b>", jdresumebrief=answer)
        else:
            response = "Please upload both a Job Description and a Resume file (PDF, DOCX or TXT)."
            return render_template("jdresume.html", result = response)
    except Exception as e:
        response = "Something went wrong. Please try again.<br><br>"+"Error is: <br/>"+str(e)
        return render_template("jdresume.html", result = response)


@app.route("/pipeline")
def pipeline():
    return render_template("pipeline.html")


@app.route("/pipeline", methods=["POST"])
def pipeline_run():
    try:
        jd_file = request.files.get("jd_file")
        resume_file = request.files.get("resume_file")
        if not (jd_file and jd_file.filename and resume_file and resume_file.filename):
            error = "Please upload both a Job Description and a Resume file (PDF, DOCX or TXT)."
            return render_template("pipeline.html", error=error)

        jd_text = extract_text_from_upload(jd_file)
        resume_text = extract_text_from_upload(resume_file)
        eval_id, context = run_pipeline(jd_text, resume_text)
        return render_template("pipeline.html", eval_id=eval_id, ctx=context.to_dict())
    except Exception as e:
        error = "Something went wrong running the pipeline. Please try again.<br><br>Error is: " + str(e)
        return render_template("pipeline.html", error=error)


@app.route("/pipeline/<eval_id>/questions", methods=["POST"])
def pipeline_questions(eval_id):
    context = get_context(eval_id)
    if context is None:
        return render_template("pipeline.html", error="This evaluation has expired. Please run the pipeline again.")
    try:
        questions_result = generate_questions(
            context.jd_analysis, context.skills_extraction,
            context.screener_result, context.experience_result
        )
        return render_template("pipeline.html", eval_id=eval_id, ctx=context.to_dict(), questions_result=questions_result)
    except Exception as e:
        error = "Could not generate questions. Please try again.<br><br>Error is: " + str(e)
        return render_template("pipeline.html", eval_id=eval_id, ctx=context.to_dict(), error=error)


@app.route("/pipeline/<eval_id>/email", methods=["POST"])
def pipeline_email(eval_id):
    context = get_context(eval_id)
    if context is None:
        return render_template("pipeline.html", error="This evaluation has expired. Please run the pipeline again.")
    try:
        candidate_name = str(request.form.get("candidate_name", ""))
        email_type = str(request.form.get("emailtype", "status update"))
        tone = str(request.form.get("tone", "professional"))
        email_result = draft_email(email_type, tone, context.jd_analysis, context.ranking_result, candidate_name)
        return render_template("pipeline.html", eval_id=eval_id, ctx=context.to_dict(), email_result=email_result)
    except Exception as e:
        error = "Could not draft the email. Please try again.<br><br>Error is: " + str(e)
        return render_template("pipeline.html", eval_id=eval_id, ctx=context.to_dict(), error=error)


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=8080)
