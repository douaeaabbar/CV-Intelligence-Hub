import streamlit as st
import time
import os

# Configuration de la page
st.set_page_config(
    page_title="CV Intelligence Hub Al Jisr",
    page_icon="https://th.bing.com/th/id/R.bcdbdbc62adeaed5609c065386fad3aa?rik=zP57WYVjjN2eUg&pid=ImgRaw&r=0",
    layout="wide",
)

# Charger le CSS
def load_css(file_name):
    css_path = os.path.join(os.path.dirname(__file__), file_name)
    with open(css_path, 'r', encoding='utf-8') as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# Initialiser l'état de session pour le carrousel
if 'image_index' not in st.session_state:
    st.session_state.image_index = 0
if 'last_update' not in st.session_state:
    st.session_state.last_update = time.time()

# Liste des images pour le carrousel
images = [
    "jisr.png",
    "jisr2.png",
    "jisr3.png"
]

# Logique du carrousel automatique (toutes les 2 secondes)
current_time = time.time()
if current_time - st.session_state.last_update >= 2:
    st.session_state.image_index = (st.session_state.image_index + 1) % len(images)
    st.session_state.last_update = current_time
    st.rerun()

# Header avec navigation
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/flag-icon-css/3.5.0/css/flag-icon.min.css">
            
    <header class="app-header">
        <div class="logo-container">
            <div class="app-logo">
                <img src="https://th.bing.com/th/id/R.bcdbdbc62adeaed5609c065386fad3aa?rik=zP57WYVjjN2eUg&pid=ImgRaw&r=0" class="logo-img">
            </div>
            <nav class="nav-menu">
                <a href="#" class="nav-item active">Accueil</a>
                <a href="#" class="nav-item">Fonctionnalités</a>
                <a href="#" class="nav-item">Comment ça marche</a>
                <a href="#" class="nav-item">Modèles</a>
                <a href="#" class="nav-item">Témoignages</a>
            </nav> 
        </div>
        <div class="right-section">
            <a href="#" class="header-icon"><i class="far fa-star"></i></a>
            <a href="#" class="header-icon"><i class="far fa-bell"></i></a>  
            <div class="flag">
                <i class="flag-icon flag-icon-ma"></i>
            </div>     
        </div>     
    </header>
""", unsafe_allow_html=True)

# Section principale avec contenu et carrousel
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    <div class="content-section">
        <div class="landing-title">
            <h1 class="main-title">CV Intelligence Hub AL Jisr</h1> 
            <p class="main-description">
                Une plateforme intelligente de traitement et d'analyse de CV, développée pour aider à l'orientation des candidats 
                vers les formations les plus adaptées. Grâce à l'IA, elle évalue automatiquement les profils, prédit leur adéquation 
                avec différentes catégories de formation, et fournit des recommandations personnalisées pour renforcer leur employabilité.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Bouton d'action principal
    if st.button(" Commencer l'Analyse CV", key="analyze_btn"):
        st.switch_page("pages/streamlit_cv_interface.py")

with col2:
    # Container pour le carrousel d'images
    st.markdown("""<div class="carousel-container">""", unsafe_allow_html=True)
    
    # Affichage de l'image actuelle
    current_image = images[st.session_state.image_index]
    st.image(current_image, use_container_width=True)
    
    # Indicateurs du carrousel
    indicators_html = '<div class="carousel-indicators">'
    for i in range(len(images)):
        active_class = "active" if i == st.session_state.image_index else ""
        indicators_html += f'<div class="indicator {active_class}"></div>'
    indicators_html += '</div>'
    
    st.markdown(indicators_html, unsafe_allow_html=True)
    st.markdown("""</div>""", unsafe_allow_html=True)

# Section des fonctionnalités
st.markdown("""
<div class="features-section">
    <h2 class="section-title">Pourquoi choisir CV Intelligence Hub ?</h2>
    <div class="features-grid">
        <div class="feature-card">
            <div class="feature-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.94-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" fill="#00C301"/>
                </svg>
            </div>
            <h3>IA Avancée</h3>
            <p>Analyse intelligente des CV avec des algorithmes de machine learning</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z" fill="#00C301"/>
                </svg>
            </div>
            <h3>Analyse Précise</h3>
            <p>Évaluation détaillée des compétences et recommandations personnalisées</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" fill="#00C301"/>
                </svg>
            </div>
            <h3>Orientation Ciblée</h3>
            <p>Suggestions de formations adaptées à votre profil professionnel</p>
        </div>
        <div class="feature-card">
            <div class="feature-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M7 2v11h3v9l7-12h-4l4-8z" fill="#00C301"/>
                </svg>
            </div>
            <h3>Résultats Rapides</h3>
            <p>Analyse instantanée et rapports détaillés en quelques secondes</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Footer amélioré
st.markdown("""
    <footer class="app-footer">
        <div class="footer-content">
            <div class="footer-left">
                <div class="footer-logo">
                    <img src="https://th.bing.com/th/id/R.bcdbdbc62adeaed5609c065386fad3aa?rik=zP57WYVjjN2eUg&pid=ImgRaw&r=0" alt="Al Jisr Logo">
                </div>
                <div class="footer-info">
                    <h4>CV Intelligence Hub</h4>
                    <p>Al Jisr - Plateforme d'analyse intelligente de CV</p>
                </div>
            </div>
            <div class="footer-right">
                <div class="social-links">
                    <a href="#"><i class="fab fa-facebook-f"></i></a>
                    <a href="#"><i class="fab fa-twitter"></i></a>
                    <a href="#"><i class="fab fa-linkedin-in"></i></a>
                    <a href="#"><i class="fab fa-instagram"></i></a>
                </div>
            </div>
        </div>
        <div class="footer-bottom">
            <p>&copy; 2025 Al Jisr - CV Intelligence Hub. Tous droits réservés.</p>
        </div>
    </footer>
""", unsafe_allow_html=True)

# Auto-refresh pour le carrousel
placeholder = st.empty()
with placeholder:
    time.sleep(0.1)
    st.rerun()
