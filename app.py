"""
Deep Space Station 9 AI Assistant - Operations Subsystem
Pure local NLP (TF-IDF + cosine similarity) - NO external API, NO API key required.
Dependencies: Flask, scikit-learn, nltk, numpy
"""
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import string

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Try to use NLTK stopwords/tokenizer; fall back to a built-in list if the
# NLTK data isn't downloaded yet, so first run never crashes.
# ---------------------------------------------------------------------------
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords as nltk_stopwords

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        try:
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

    STOP_WORDS = set(nltk_stopwords.words('english'))
    USE_NLTK = True
except Exception:
    STOP_WORDS = set("""a an the is are was were be been being have has had do does did
        will would shall should may might must can could of in on at to for with by from
        as it this that these those i you he she we they my your his her its our their
        not no so if then than but and or""".split())
    USE_NLTK = False


def tokenize(text):
    if USE_NLTK:
        try:
            return word_tokenize(text)
        except Exception:
            pass
    return re.findall(r"[a-zA-Z']+", text)


def preprocess_text(text):
    """Clean and normalize user input for matching."""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = tokenize(text)
    tokens = [w for w in tokens if w not in STOP_WORDS]
    return ' '.join(tokens) if tokens else text


# ---------------------------------------------------------------------------
# FAQ Knowledge Base - 52 Q&A pairs across 7 categories
# ---------------------------------------------------------------------------
FAQS = {
    # Core Operations
    "How do I activate artificial gravity?": "Gravity systems are controlled via Engineering Panel 7-A. Current gravity: 1.2G. Toggle switch sequence: Power > Calibrate > Engage.",
    "What's the current station status?": "All systems nominal. Life support: 100%. Power: 98%. Hull integrity: 100%.",
    "How do I access the main computer?": "Use any terminal and authenticate with your crew ID badge. Voice command 'Computer, identify' also works.",
    "What's the crew duty roster?": "Duty roster is posted on the mess hall display and synced to your personal terminal each morning at 0600.",
    "How do I schedule maintenance?": "Open the Maintenance Request app on your terminal, select the system, and choose an available time slot.",
    "Where's the command center?": "Command center is on Deck 1, forward section, accessible via the central lift.",
    "How do I adjust life support?": "Life support controls are on Engineering Panel 3-B. Only certified personnel should adjust O2/CO2 mix ratios.",
    "What's the current power status?": "Reactor output is at 98% capacity. Backup batteries fully charged. No power conservation needed.",
    "How do I use the environmental controls?": "Environmental controls are accessible from any cabin terminal under the 'Habitat' menu - adjust temperature, humidity, and lighting.",
    "What's today's schedule?": "Today's schedule includes morning systems check, crew exercise block, and afternoon research session. Check your terminal for exact times.",

    # Emergency Procedures
    "What should I do if I detect an unknown object?": "ALERT PROTOCOL: 1) Activate sensors and log trajectory 2) Contact Command immediately 3) Initiate red alert if object is on collision course or hostile.",
    "How do I sound the red alert?": "Press the red alert panel near any bulkhead, or say 'Computer, red alert' to any terminal. This notifies the entire crew instantly.",
    "How do I seal a hull breach?": "Activate emergency bulkhead seal via the nearest red panel. Emergency force fields engage automatically within 3 seconds of breach detection.",
    "What's the emergency evacuation procedure?": "Proceed to your assigned escape pod bay. Pods launch automatically 60 seconds after the evacuation order unless manually held.",
    "How do I use the emergency override?": "Emergency override requires a command-level keycard. Insert into any red-panel terminal and follow voice prompts.",
    "What should I do during a solar flare?": "Move to shielded sections (marked with radiation symbols). Avoid EVA activity. Communication blackout expected for 15-45 minutes.",
    "How do I activate emergency shields?": "Shields activate automatically on red alert, or manually via the Defense Console on the Bridge.",
    "What's the fire suppression protocol?": "Fire suppression triggers automatically via oxygen displacement in the affected compartment. Evacuate that section immediately and seal the door behind you.",
    "How do I eject cargo in emergency?": "Cargo bay has manual jettison levers near each airlock, or remote jettison via the Cargo Management terminal.",
    "What to do if systems fail?": "Switch to backup power immediately, notify Command, and consult the System Failure checklist in your emergency manual app.",

    # Communications
    "How do I contact Earth Command?": "Use the quantum comm array on the Bridge. Current signal delay to Earth: 8.3 minutes one-way.",
    "Can we communicate during solar flare?": "Standard comms are disrupted during solar flares. Quantum-encrypted backup channel remains operational but with reduced bandwidth.",
    "How do I send encrypted messages?": "Select 'Secure Transmission' on the comm terminal and enter your encryption key. Messages are AES-256 encrypted by default.",
    "What's the backup communication system?": "Backup laser-relay communication system, located in the auxiliary comm bay on Deck 2.",
    "How long to reach Earth?": "At current quantum-relay speeds, message delay to Earth is approximately 8.3 minutes one-way.",
    "How do I reach nearby stations?": "Use the short-range subspace radio on the Bridge - effective range is approximately 2 light-minutes.",
    "Can aliens intercept our messages?": "All outgoing transmissions use quantum encryption, making interception statistically near-impossible with known technology.",
    "How do I boost signal strength?": "Increase comm array power allocation via Engineering Panel 5-C, up to a maximum safe output of 150%.",

    # Medical
    "Where's the medical bay?": "Medical bay is on Deck 2, mid-section, adjacent to the crew quarters.",
    "How do I treat radiation exposure?": "Administer the radiation treatment kit from the medical bay immediately and report to the medical officer for full decontamination protocol.",
    "What's the crew health status?": "All crew members are currently reporting healthy vitals according to the latest biometric scan.",
    "How do I perform CPR in zero gravity?": "Anchor yourself and the patient using the restraint straps, then apply chest compressions at the standard rate using your full body weight for leverage.",
    "Where are the medical supplies?": "Medical supplies are stored in the cabinets along the medical bay's east wall, organized by category.",
    "How long does healing take in space?": "Wound healing in microgravity can take 20-30% longer than on Earth due to reduced circulation; the medical bay has accelerated healing equipment to compensate.",
    "What's the quarantine protocol?": "Suspected contamination cases are isolated in the quarantine pod on Deck 2 for a minimum 72-hour observation period.",
    "How do I treat decompression sickness?": "Move the patient to the hyperbaric chamber in the medical bay immediately and notify the medical officer.",

    # Alien Contact
    "What should I do if I detect alien life?": "Do not engage. Log all sensor data, notify Command immediately, and prepare for First Contact Protocol if instructed.",
    "What's the first contact protocol?": "First Contact Protocol: 1) Maintain distance 2) Attempt only passive communication 3) Record everything 4) Await Command authorization before direct contact.",
    "Are there hostile species nearby?": "No confirmed hostile species detected in current sector based on the latest long-range scans.",
    "How do I communicate with aliens?": "Use the universal translator module on the Bridge - it analyzes signal patterns and attempts real-time linguistic mapping.",
    "What was that unidentified signal?": "Unidentified signal has been logged and forwarded to the Signals Analysis team for pattern matching against known databases.",
    "How do I analyze alien technology?": "Bring any recovered technology to the Science Lab for non-invasive scanning. Never attempt direct power-up without lab clearance.",

    # Navigation
    "What's our current orbit?": "Current orbit is stable at 400km above the reference body, with a period of approximately 92 minutes.",
    "Are we on collision course?": "No collision course detected. Navigation systems continuously scan for debris and adjust trajectory as needed.",
    "How do I adjust our trajectory?": "Trajectory adjustments are made from the Navigation Console on the Bridge using the thruster control interface.",
    "What's our distance from Mars?": "Current distance from Mars varies by orbital position - check the live readout on the Navigation Console for exact figures.",
    "Can we reach the moon from here?": "Yes, a lunar transfer burn would take approximately 3 days at standard thrust profiles.",

    # General
    "When's the next supply ship?": "Next supply ship is scheduled to arrive in accordance with the resupply calendar - check the Logistics terminal for the exact ETA.",
    "What's the crew meal schedule?": "Meals are served at 0700, 1200, and 1800 station time in the mess hall.",
    "Where's the exercise area?": "Exercise area is on Deck 3, equipped with resistance machines and a treadmill calibrated for variable gravity.",
    "How long until next communication window?": "Communication windows with Earth open every 6 hours, synced to orbital positioning. Check the Comm Console for the next countdown.",
    "What's the training schedule?": "Training drills (fire, depressurization, medical) are scheduled weekly - check your personal terminal calendar for assignments.",
}

EMERGENCY_KEYWORDS = ['emergency', 'critical', 'help', 'urgent', 'breach', 'fire', 'attack', 'mayday']

faq_questions = list(FAQS.keys())
faq_answers = list(FAQS.values())

vectorizer = TfidfVectorizer(preprocessor=preprocess_text)
faq_vectors = vectorizer.fit_transform(faq_questions)


def find_best_matches(user_question, top_n=3):
    """Find the best matching FAQ(s) using TF-IDF + cosine similarity."""
    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, faq_vectors)[0]
    ranked_indices = similarities.argsort()[::-1][:top_n]
    return [(idx, float(similarities[idx])) for idx in ranked_indices]


def is_emergency(text):
    lowered = text.lower()
    return any(kw in lowered for kw in EMERGENCY_KEYWORDS)


@app.route('/')
def index():
    return render_template('chat.html')


@app.route('/ask', methods=['POST'])
def ask_assistant():
    data = request.json or {}
    question = (data.get('question') or '').strip()

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    matches = find_best_matches(question, top_n=3)
    best_idx, best_score = matches[0]
    confidence_pct = round(best_score * 100, 1)
    emergency = is_emergency(question)

    if best_score < 0.15:
        answer = "I'm not certain about that. Try rephrasing your question, or contact Command directly for assistance."
        matched_q = None
        confidence_pct = 0
        related = []
    else:
        matched_q = faq_questions[best_idx]
        answer = faq_answers[best_idx]
        related = [
            {'question': faq_questions[idx], 'confidence': round(score * 100, 1)}
            for idx, score in matches[1:] if score > 0.1
        ]

    if confidence_pct >= 80:
        confidence_level = 'high'
    elif confidence_pct >= 40:
        confidence_level = 'medium'
    else:
        confidence_level = 'low'

    return jsonify({
        'success': True,
        'answer': answer,
        'matched_question': matched_q,
        'confidence': confidence_pct,
        'confidence_level': confidence_level,
        'user_question': question,
        'emergency': emergency,
        'related_questions': related
    })


@app.route('/faqs', methods=['GET'])
def get_all_faqs():
    category_map = {
        "Core Operations": faq_questions[0:10],
        "Emergency Procedures": faq_questions[10:20],
        "Communications": faq_questions[20:28],
        "Medical": faq_questions[28:36],
        "Alien Contact": faq_questions[36:42],
        "Navigation": faq_questions[42:47],
        "General": faq_questions[47:52],
    }
    return jsonify({
        'faqs': [{'q': q, 'a': a} for q, a in FAQS.items()],
        'categories': category_map,
        'total': len(FAQS)
    })


if __name__ == '__main__':
    print("Deep Space Station 9 AI Assistant starting...")
    print(f"Loaded {len(FAQS)} FAQs across 7 categories")
    print("Using local TF-IDF + cosine similarity (no external API)")
    print("Running at http://localhost:5000")
    app.run(debug=True, port=5000)