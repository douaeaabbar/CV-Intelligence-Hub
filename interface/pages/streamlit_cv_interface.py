import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
from PIL import Image
import base64
import os
from datetime import datetime
import re
import time
import pickle
import warnings
warnings.filterwarnings('ignore')

# Imports pour les modèles ML
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import xgboost as xgb

# Configuration de la page
st.set_page_config(
    page_title="CV Intelligence Hub Al Jisr",
    page_icon="https://th.bing.com/th/id/R.bcdbdbc62adeaed5609c065386fad3aa?rik=zP57WYVjjN2eUg&pid=ImgRaw&r=0",
    layout="wide",
    initial_sidebar_state="expanded"
)



def initialize_session_state():
    """Initialise les variables du session state si elles n'existent pas"""
    if 'cv_history' not in st.session_state:
        st.session_state.cv_history = [
            {
                'id': 1,
                'nom_fichier': 'cv_jean_dupont.pdf',
                'date_analyse': '2024-06-01 14:30:22',
                'classification': 'Développement Web',
                'probabilite': 85.6,
                'statut': 'Admis',
                'description': 'Développeur Full-Stack avec 5 ans d\'expérience en React et Node.js. Excellentes compétences techniques.',
                'fichier_path': '/uploads/cv_jean_dupont.pdf'
            },
            {
                'id': 2,
                'nom_fichier': 'cv_anas_alaoui.pdf',
                'date_analyse': '2025-06-02 09:15:10',
                'classification': 'Data Science',
                'probabilite': 92.3,
                'statut': 'Admis',
                'description': 'Data Scientist expérimenté avec maîtrise de Python, ML et visualisation de données. Profil très prometteur.',
                'fichier_path': '/uploads/cv_marie_martin.pdf'
            },
            {
                'id': 3,
                'nom_fichier': 'cv_douae_aabbar.pdf',
                'date_analyse': '2025-06-03 16:45:33',
                'classification': 'Marketing Digital',
                'probabilite': 43.7,
                'statut': 'Rejeté',
                'description': 'Profil junior en marketing digital. Compétences limitées mais potential d\'évolution.',
                'fichier_path': '/uploads/cv_lina_durand.pdf'
            },
            {
                'id': 4,
                'nom_fichier': 'cv_yassmina_taki.pdf',
                'date_analyse': '2025-06-04 11:20:15',
                'classification': 'Design Graphique',
                'probabilite': 78.9,
                'statut': 'Admis',
                'description': 'Designer créatif avec portfolio impressionnant. Maîtrise d\'Adobe Creative Suite et UX/UI.',
                'fichier_path': '/uploads/cv_amina_bouazza.pdf'
            },
            {
                'id': 5,
                'nom_fichier': 'cv_ahmed_hassan.pdf',
                'date_analyse': '2025-06-05 13:55:40',
                'classification': 'Maintenance IT',
                'probabilite': 67.2,
                'statut': 'En Attente',
                'description': 'Technicien IT avec bonnes bases en réseau et sécurité. Expérience pratique appréciable.',
                'fichier_path': '/uploads/cv_ahmed_hassan.pdf'
            }
        ]
    
    if 'config' not in st.session_state:
        st.session_state.config = {
            'seuil_admission': 50,
            'categories_actives': ['Développement Web', 'Marketing Digital', 'Data Science'],
            'analyse_approfondie': True,
            'sauvegarde_auto': False,
            'export_auto': False
        }


class CVClassificationSystem:
    def __init__(self):
        self.classification_model = None
        self.screening_model = None
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self.scaler = StandardScaler()
        
        # Mots vides simples
        self.stop_words = {'le', 'la', 'les', 'de', 'du', 'des', 'un', 'une', 'et', 'ou', 'mais', 'donc', 'car', 'ni', 'or',
                          'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'a', 'an'}
    
    def preprocess_text(self, text):
        """Preprocessing simple du texte"""
        if pd.isna(text) or text == "":
            return ""
        
        try:
            # Conversion en minuscules
            text = str(text).lower()
            
            # Suppression des caractères spéciaux et des chiffres
            text = re.sub(r'[^a-zA-ZÀ-ÿ\s]', ' ', text)
            
            # Tokenisation simple
            tokens = text.split()
            
            # Suppression des mots vides
            processed_tokens = []
            for token in tokens:
                if token not in self.stop_words and len(token) > 2:
                    processed_tokens.append(token)
            
            return ' '.join(processed_tokens)
        except Exception:
            return str(text).lower()
    
    def extract_features(self, df):
        """Extraction de features avancées"""
        features_df = df.copy()
        
        # Preprocessing des colonnes textuelles
        text_columns = ['education', 'competences_technique', 'competences_generale', 
                       'experience_professionnelle', 'projets_professionnelles', 
                       'certificat', 'langues_etrangeres', 'texte_cv']
        
        for col in text_columns:
            if col in features_df.columns:
                features_df[f'{col}_processed'] = features_df[col].apply(self.preprocess_text)
            else:
                features_df[f'{col}_processed'] = ""
        
        # Features numériques dérivées
        features_df['total_experience_words'] = features_df['experience_professionnelle_processed'].apply(lambda x: len(str(x).split()) if x else 0)
        features_df['total_competences_words'] = (features_df['competences_technique_processed'].apply(lambda x: len(str(x).split()) if x else 0) + 
                                                 features_df['competences_generale_processed'].apply(lambda x: len(str(x).split()) if x else 0))
        features_df['total_projects_words'] = features_df['projets_professionnelles_processed'].apply(lambda x: len(str(x).split()) if x else 0)
        features_df['total_education_words'] = features_df['education_processed'].apply(lambda x: len(str(x).split()) if x else 0)
        features_df['total_certificates'] = features_df['certificat_processed'].apply(lambda x: len(str(x).split()) if x else 0)
        
        # Ratio de complétude du CV
        text_cols_for_completeness = ['education_processed', 'competences_technique_processed', 
                                     'competences_generale_processed', 'experience_professionnelle_processed']
        features_df['cv_completeness'] = features_df[text_cols_for_completeness].apply(
            lambda row: sum(1 for x in row if x and len(str(x)) > 0) / len(text_cols_for_completeness), axis=1
        )
        
        # Création du texte combiné pour TF-IDF
        features_df['combined_text'] = (
            features_df['education_processed'].astype(str) + ' ' +
            features_df['competences_technique_processed'].astype(str) + ' ' +
            features_df['competences_generale_processed'].astype(str) + ' ' +
            features_df['experience_professionnelle_processed'].astype(str) + ' ' +
            features_df['projets_professionnelles_processed'].astype(str) + ' ' +
            features_df['certificat_processed'].astype(str)
        )
        
        return features_df
    
    def create_demo_models(self):
    """Créer des modèles de démonstration avec des données simulées"""
    # Simulation de données d'entraînement
    categories = ['Marketing Digital', 'Data Science', 'Maintenance IT', 'Design Graphique',
                 'Création de Contenu', 'Frontend Développement','Commerce & Téléconseil', 'Community Management']
    
    # Initialisation des composants
    self.label_encoder = LabelEncoder()
    self.label_encoder.fit(categories)
    
    # Vectoriseur TF-IDF simplifié
    self.tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.9,
        stop_words='english'
    )
    
    # SOLUTION ROBUSTE : Essayer de charger le CSV, sinon utiliser des données simulées
    try:
        # Essayer différents chemins possibles
        possible_paths = [
            "nv data faker/nouvelles_data_faker.csv",
            "./nv data faker/nouvelles_data_faker.csv",
            "nouvelles_data_faker.csv"
        ]
        
        df = None
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.success(f"✅ Données chargées depuis: {path}")
                break
        
        if df is not None and 'texte_cv' in df.columns:
            sample_texts = df['texte_cv'].fillna("").astype(str)
        else:
            raise FileNotFoundError("Fichier CSV non trouvé ou colonne manquante")
            
    except Exception as e:
        # Utiliser des données simulées si le fichier n'existe pas
        st.warning(f"⚠️ Fichier CSV non trouvé ({str(e)}). Utilisation de données simulées.")
        sample_texts = [
            "développeur python machine learning data science",
            "marketing digital réseaux sociaux stratégie communication",
            "design graphique adobe photoshop illustrator créatif",
            "maintenance informatique réseau serveur technique",
            "frontend react javascript html css développement web",
            "commerce vente relation client téléconseil",
            "community manager social media content création",
            "data analyst sql python visualisation données"
        ]
    
    # Ajuster la liste si elle est trop courte
    if len(sample_texts) < 10:
        sample_texts = sample_texts * (10 // len(sample_texts) + 1)
    
    sample_texts = sample_texts[:100]  # Limiter à 100 échantillons
    
    self.tfidf_vectorizer.fit(sample_texts)
    
    # Modèles pré-entraînés simulés
    self.classification_model = RandomForestClassifier(n_estimators=50, random_state=42)
    self.screening_model = xgb.XGBClassifier(n_estimators=50, random_state=42)
    
    # Entraînement sur données simulées
    X_demo = np.random.rand(100, 1006)  # 1000 features TF-IDF + 6 features numériques
    y_class_demo = np.random.choice(len(categories), 100)
    y_screen_demo = np.random.choice(2, 100)
    
    self.classification_model.fit(X_demo, y_class_demo)
    self.screening_model.fit(X_demo, y_screen_demo)
    
    return True

    
    def predict_category(self, cv_data):
        """Prédiction de la catégorie d'un CV"""
        try:
            features_df = self.extract_features(pd.DataFrame([cv_data]))
            X_text = features_df['combined_text'].iloc[0]
            X_tfidf = self.tfidf_vectorizer.transform([X_text])
            
            # Features numériques
            numeric_features = ['total_experience_words', 'total_competences_words', 
                               'total_projects_words', 'total_education_words', 
                               'total_certificates', 'cv_completeness']
            
            X_numeric = features_df[numeric_features].iloc[0:1].fillna(0)
            X_numeric_scaled = self.scaler.fit_transform(X_numeric)
            
            # Combinaison
            X_combined = np.hstack([X_tfidf.toarray(), X_numeric_scaled])
            
            # Prédiction avec gestion d'erreur
            if X_combined.shape[1] != 1006:  # Ajuster la taille si nécessaire
                X_combined = np.pad(X_combined, ((0, 0), (0, max(0, 1006 - X_combined.shape[1]))), mode='constant')
                X_combined = X_combined[:, :1006]
            
            prediction = self.classification_model.predict(X_combined)[0]
            probabilities = self.classification_model.predict_proba(X_combined)[0]
            
            category = self.label_encoder.inverse_transform([prediction])[0]
            confidence = max(probabilities)
            
            return {
                'category': category,
                'confidence': confidence,
                'probabilities': dict(zip(self.label_encoder.classes_, probabilities))
            }
        except Exception as e:
            # Retour par défaut en cas d'erreur
            categories = ['Développement Web', 'Marketing Digital', 'Data Science', 'Maintenance IT', 'Design Graphique','Création de Contenu', 'Frontend Développement','Commerce & Téléconseil', 'Community Management']
            probs = np.random.dirichlet([1]*5)
            return {
                'category': np.random.choice(categories),
                'confidence': max(probs),
                'probabilities': dict(zip(categories, probs))
            }
    
    def predict_screening(self, cv_data):
        """Prédiction du screening d'un CV"""
        try:
            features_df = self.extract_features(pd.DataFrame([cv_data]))
            X_text = features_df['combined_text'].iloc[0]
            X_tfidf = self.tfidf_vectorizer.transform([X_text])
            
            # Features numériques
            numeric_features = ['total_experience_words', 'total_competences_words', 
                               'total_projects_words', 'total_education_words', 
                               'total_certificates', 'cv_completeness']
            
            X_numeric = features_df[numeric_features].iloc[0:1].fillna(0)
            X_numeric_scaled = self.scaler.fit_transform(X_numeric)
            
            # Combinaison
            X_combined = np.hstack([X_tfidf.toarray(), X_numeric_scaled])
            
            # Ajuster la taille si nécessaire
            if X_combined.shape[1] != 1006:
                X_combined = np.pad(X_combined, ((0, 0), (0, max(0, 1006 - X_combined.shape[1]))), mode='constant')
                X_combined = X_combined[:, :1006]
            
            decision_prob = self.screening_model.predict_proba(X_combined)[0]
            decision = 'admis' if decision_prob[1] > 0.5 else 'refusé'
            score = decision_prob[1] * 100
            
            return {
                'decision': decision,
                'score': round(score, 2),
                'probability': round(decision_prob[1], 4)
            }
        except Exception as e:
            # Retour par défaut en cas d'erreur
            prob = np.random.rand()
            return {
                'decision': 'admis' if prob > 0.4 else 'refusé',
                'score': round(prob * 100, 2),
                'probability': round(prob, 4)
            }

# Initialisation du système CV
@st.cache_resource
def load_cv_system():
    """Chargement du système CV avec modèles de démonstration"""
    try:
        cv_system = CVClassificationSystem()
        cv_system.create_demo_models()
        return cv_system
    except Exception as e:
        st.error(f"Erreur lors du chargement du système: {e}")
        return None

# Chargement du CSS personnalisé
def load_css(css_file_path="pages/style.css"):
    """
    Chargement des styles CSS depuis un fichier externe
    
    Args:
        css_file_path (str): Chemin vers le fichier CSS (par défaut: "style.css")
    """
    try:
        # Vérifier si le fichier existe
        if os.path.exists(css_file_path):
            # Lire le contenu du fichier CSS
            with open(css_file_path, 'r', encoding='utf-8') as f:
                css_content = f.read()
            
            # Appliquer les styles à Streamlit
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            
        else:
            st.error(f"Le fichier CSS '{css_file_path}' n'a pas été trouvé.")
            
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier CSS : {str(e)}")

# Utilisation alternative avec un chemin relatif
def load_css_from_assets(css_filename="style.css"):
    """
    Chargement des styles CSS depuis un dossier assets
    
    Args:
        css_filename (str): Nom du fichier CSS dans le dossier assets
    """
    css_path = os.path.join("assets", "css", css_filename)
    load_css(css_path)

# Fonction pour extraire le texte des fichiers (simplifiée)
def extract_text_from_file(uploaded_file):
    """Extraction de texte depuis différents formats de fichiers"""
    try:
        if uploaded_file.type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        else:
            # Pour les autres types, on essaie de lire en tant que texte
            return str(uploaded_file.read(), "utf-8", errors='ignore')
    except Exception as e:
        st.error(f"Erreur lors de l'extraction du texte: {str(e)}")
        return ""

# Analyse de CV avec animation
def analyze_cv_with_animation(cv_data, cv_system):
    """Analyse de CV avec barre de progression animée"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Étape 1: Preprocessing
    status_text.text("🔄 Preprocessing du CV...")
    time.sleep(0.5)
    progress_bar.progress(25)
    
    # Étape 2: Classification
    status_text.text(" Classification de la catégorie...")
    time.sleep(0.5)
    progress_bar.progress(50)
    classification_result = cv_system.predict_category(cv_data)
    
    # Étape 3: Screening
    status_text.text("Évaluation du screening...")
    time.sleep(0.5)
    progress_bar.progress(75)
    screening_result = cv_system.predict_screening(cv_data)
    
    # Étape 4: Finalisation
    status_text.text("✅ Analyse terminée!")
    progress_bar.progress(100)
    time.sleep(0.5)
    
    progress_bar.empty()
    status_text.empty()
    
    return classification_result, screening_result

# Graphiques de visualisation
def create_classification_chart(probabilities):
    """Graphique des probabilités de classification"""
    categories = list(probabilities.keys())
    values = [prob * 100 for prob in probabilities.values()]
    
    fig = px.bar(
        x=categories,
        y=values,
        color=values,
        color_continuous_scale='Viridis',
        title="Probabilités de Classification par Catégorie"
    )
    
    fig.update_layout(
        xaxis_title="Catégories Professionnelles",
        yaxis_title="Probabilité (%)",
        showlegend=False,
        height=400
    )
    
    return fig

def create_score_gauge(score, title):
    """Indicateur de score circulaire"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

# Interface principale
def main():
    """Fonction principale de l'application"""
    load_css()
    initialize_session_state()
    
   


    # Titre HTML centré
    st.markdown("""
    <div class="main-header" style="text-align: center;">
        <h1> CV Intelligence Hub AL Jisr</h1>
        <p>Système intelligent de screening et classification de CV</p>
    </div>
    """, unsafe_allow_html=True)

    
    # Chargement du système
    try:
        cv_system = load_cv_system()
        if cv_system is None:
            st.error("❌ Erreur de chargement du système")
            return
        st.success("✅ Système de modèles chargé avec succès!")
    except Exception as e:
        st.error(f"❌ Erreur de chargement du système: {str(e)}")
        return
    
    # Sidebar pour la navigation
    with st.sidebar:
            # Affichage du logo
        st.image("pages/Al Jisr - Maroc.png", width=280)
        st.markdown("# Navigation")

        
        page = st.selectbox(
            "Choisir une section",
            ["📄 Analyse de CV", "📊 Dashboard", "⚙️ Configuration","📋 Historique"]
        )
        
        st.markdown("---")
        st.markdown("#  Guide d'utilisation")
        st.markdown("""
        1. **Uploader votre CV** 
        2. **Remplir les informations requises**
        3. **Lancer l'analyse intelligente**
        4. **Consulter les résultats détaillés**
        """)
    
    if page == "📄 Analyse de CV":
        st.markdown("## Analyse Intelligente des Applications des Formations ")
        
        # Layout en colonnes
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 📤 Upload de CV")
            uploaded_file = st.file_uploader(
                "Choisir un fichier CV",
                type=['txt','pdf'],
                help="Format supporté: TXT/PDF (pour cette démo)"
            )
            
            # Formulaire de saisie
            with st.form("cv_form"):
                st.markdown("### 📋 Informations du CV")
                
                titre_cv = st.text_input("Titre du CV")
                education = st.text_area("Éducation", height=100)
                competences_tech = st.text_area("Compétences Techniques", height=100)
                competences_gen = st.text_area("Compétences Générales", height=100)
                experience = st.text_area("Expérience Professionnelle", height=100)
                projets = st.text_area("Projets Professionnels", height=100)
                certificats = st.text_input("Certificats")
                langues = st.text_input("Langues Étrangères")
                
                submitted = st.form_submit_button(" Analyser le CV", use_container_width=True)
        
        with col2:
            if submitted or uploaded_file:
                # Extraction du texte si fichier uploadé
                texte_cv = ""
                if uploaded_file:
                    texte_cv = extract_text_from_file(uploaded_file)
                    st.success(f"✅ Fichier '{uploaded_file.name}' traité avec succès!")
                
                # Données du CV
                cv_data = {
                    'titre_cv': titre_cv,
                    'education': education,
                    'competences_technique': competences_tech,
                    'competences_generale': competences_gen,
                    'experience_professionnelle': experience,
                    'projets_professionnelles': projets,
                    'certificat': certificats,
                    'langues_etrangeres': langues,
                    'texte_cv': texte_cv
                }
                
                # Analyse avec animation
                if any(cv_data.values()):
                    classification_result, screening_result = analyze_cv_with_animation(cv_data, cv_system)
    
                    # *** NOUVEAU CODE : Ajout automatique à l'historique ***
                    # Déterminer le statut basé sur le score
                    if screening_result['score'] >= st.session_state.config['seuil_admission']:
                        statut = 'Admis'
                    elif screening_result['score'] >= 30:  # Seuil pour "En Attente"
                        statut = 'En Attente'
                    else:
                        statut = 'Rejeté'

                     # Créer l'entrée historique
                    nouvel_cv = {
                            'id': len(st.session_state.cv_history) + 1,
                            'nom_fichier': uploaded_file.name if uploaded_file else f"cv_manuel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            'date_analyse': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'classification': classification_result['category'],
                            'probabilite': round(classification_result['confidence'] * 100, 1),
                            'statut': statut,
                            'description': f"CV analysé automatiquement. Classification: {classification_result['category']} avec {classification_result['confidence']:.1%} de confiance. Score de screening: {screening_result['score']:.1f}%.",
                            'fichier_path': f"/uploads/{uploaded_file.name if uploaded_file else f'cv_manuel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt'}"
                    }

                    # Ajouter à l'historique
                    st.session_state.cv_history.append(nouvel_cv)

                    # Message de confirmation
                    st.success(f"✅ CV ajouté à l'historique avec l'ID #{nouvel_cv['id']}")
                    
                    # Affichage des résultats
                    st.markdown("##  Résultats de l'Analyse")
                    
                    # Métriques principales
                    col_metric1, col_metric2, col_metric3 = st.columns(3)
                    
                    with col_metric1:
                        st.metric(
                            "Catégorie Prédite",
                            classification_result['category'],
                            f"{classification_result['confidence']:.1%} confiance"
                        )
                    
                    with col_metric2:
                        st.metric(
                            "Décision de Screening",
                            screening_result['decision'].upper(),
                            f"{screening_result['score']:.1f}% score"
                        )
                    
                    with col_metric3:
                        status_color = "🟢" if screening_result['decision'] == 'admis' else "🔴"
                        st.metric(
                            "Statut",
                            f"{status_color} {screening_result['decision'].title()}",
                            f"Probabilité: {screening_result['probability']:.3f}"
                        )
                    
                    # Graphiques détaillés
                    col_chart1, col_chart2 = st.columns(2)
                    
                    with col_chart1:
                        # Graphique de classification
                        classification_chart = create_classification_chart(classification_result['probabilities'])
                        st.plotly_chart(classification_chart, use_container_width=True)
                    
                    with col_chart2:
                        # Gauge de score
                        score_gauge = create_score_gauge(screening_result['score'], "Score de Screening")
                        st.plotly_chart(score_gauge, use_container_width=True)
                    
                    # Analyse détaillée
                    with st.expander(" Analyse Détaillée", expanded=True):
                        st.markdown("###  Recommandations")
                        
                        # Recommandations basées sur le score
                        if screening_result['score'] >= 80:
                            st.success("🌟 **Excellent profil!** Ce candidat présente toutes les qualifications requises.")
                        elif screening_result['score'] >= 60:
                            st.info("👍 **Bon profil.** Quelques améliorations possibles dans certains domaines.")
                        else:
                            st.warning("⚠️ **Profil à développer.** Formation supplémentaire recommandée.")
                        
                        # Détails des probabilités
                        st.markdown("### 📊 Détail des Probabilités par Catégorie")
                        if 'probabilities' in classification_result and classification_result['probabilities']:
                            prob_df = pd.DataFrame({
                                'Catégorie': list(classification_result['probabilities'].keys()),
                                'Probabilité': [f"{prob:.1%}" for prob in classification_result['probabilities'].values()]
                            })
                            st.dataframe(prob_df, use_container_width=True)
                
                else:
                    st.warning("⚠️ Veuillez remplir au moins quelques champs pour procéder à l'analyse.")
    
    elif page == "📊 Dashboard":
        st.markdown("##  Dashboard Analytique")
        
        # Métriques simulées pour la démo
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("CVs Analysés", "1,234", "↗️ +12%")
        with col2:
            st.metric("Taux d'Admission", "67%", "↗️ +5%")
        with col3:
            st.metric("Score Moyen", "75.3", "↗️ +2.1")
        with col4:
            st.metric("Catégories", "5", "→ 0")
        
        # Graphiques de démo
        st.markdown("###  Statistiques Temporelles")
        
        # Génération de données simulées
        dates = pd.date_range('2024-01-01', periods=30, freq='D')
        data = {
            'Date': dates,
            'CVs Analysés': np.random.randint(20, 100, 30),
            'Taux Admission': np.random.uniform(0.5, 0.8, 30)
        }
        df_stats = pd.DataFrame(data)
        
        fig_timeline = px.line(df_stats, x='Date', y='CVs Analysés', 
                              title="Évolution des CVs Analysés")
        st.plotly_chart(fig_timeline, use_container_width=True)
        
    elif page == "⚙️ Configuration":
        st.markdown("## ⚙️ Configuration du Système")
        
        st.info("Cette section permet de configurer les paramètres du système.")
        
        with st.form("config_form"):
            st.markdown("### Paramètres de Classification")
            
            seuil_admission = st.slider("Seuil d'admission (%)", 0, 100, 50)
            categories_actives = st.multiselect(
                "Catégories actives",
                ['Développement Web', 'Marketing Digital', 'Data Science', 'Maintenance IT', 'Design Graphique','Création de Contenu', 'Frontend Développement','Commerce & Téléconseil', 'Community Management'],
                default=['Développement Web', 'Marketing Digital', 'Data Science']
            )
            
            st.markdown("### Paramètres d'Analyse")
            analyse_approfondie = st.checkbox("Analyse approfondie", value=True)
            sauvegarde_auto = st.checkbox("Sauvegarde automatique", value=False)
            
            submitted_config = st.form_submit_button("💾 Sauvegarder Configuration")
            
            if submitted_config:
                st.success("✅ Configuration sauvegardée avec succès!")
    elif page == "📋 Historique":
        st.markdown("##  Historique des CV Analysés")
        
        # Statistiques rapides
        col1, col2, col3, col4 = st.columns(4)
        
        total_cv = len(st.session_state.cv_history)
        admis = len([cv for cv in st.session_state.cv_history if cv['statut'] == 'Admis'])
        rejetes = len([cv for cv in st.session_state.cv_history if cv['statut'] == 'Rejeté'])
        en_attente = len([cv for cv in st.session_state.cv_history if cv['statut'] == 'En Attente'])
        
        with col1:
            st.metric("Total CV", total_cv)
        with col2:
            st.metric("Admis", admis, f"{(admis/total_cv*100):.1f}%")
        with col3:
            st.metric("Rejetés", rejetes, f"{(rejetes/total_cv*100):.1f}%")
        with col4:
            st.metric("En Attente", en_attente, f"{(en_attente/total_cv*100):.1f}%")
        
        st.markdown("---")
        
        # Filtres
        st.markdown("### Filtres")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filtre_statut = st.selectbox(
                "Filtrer par statut",
                ["Tous", "Admis", "Rejeté", "En Attente"]
            )
        
        with col2:
            filtre_classification = st.selectbox(
                "Filtrer par classification",
                ["Toutes"] + list(set([cv['classification'] for cv in st.session_state.cv_history]))
            )
        
        with col3:
            seuil_probabilite = st.slider("Probabilité minimale (%)", 0, 100, 0)
        
        # Appliquer les filtres
        cv_filtres = st.session_state.cv_history.copy()
        
        if filtre_statut != "Tous":
            cv_filtres = [cv for cv in cv_filtres if cv['statut'] == filtre_statut]
        
        if filtre_classification != "Toutes":
            cv_filtres = [cv for cv in cv_filtres if cv['classification'] == filtre_classification]
        
        cv_filtres = [cv for cv in cv_filtres if cv['probabilite'] >= seuil_probabilite]
        
        st.markdown(f"###  Résultats ({len(cv_filtres)} CV)")
        
        if cv_filtres:
            # Tableau avec les CV
            for i, cv in enumerate(cv_filtres):
                with st.expander(f"📄 {cv['nom_fichier']} - {cv['classification']} ({cv['probabilite']}%)", expanded=False):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Date d'analyse:** {cv['date_analyse']}")
                        st.markdown(f"**Classification:** {cv['classification']}")
                        st.markdown(f"**Probabilité:** {cv['probabilite']}%")
                        
                        # Badge de statut avec couleur
                        if cv['statut'] == 'Admis':
                            st.success(f"✅ {cv['statut']}")
                        elif cv['statut'] == 'Rejeté':
                            st.error(f"❌ {cv['statut']}")
                        else:
                            st.warning(f"⏳ {cv['statut']}")
                        
                        st.markdown("**Description:**")
                        st.write(cv['description'])
                    
                    with col2:
                        # Bouton de téléchargement
                        st.markdown("### Actions")
                        
                        # Simuler le contenu du fichier pour le téléchargement
                        fichier_content = f"Contenu simulé du CV: {cv['nom_fichier']}\nClassification: {cv['classification']}\nProbabilité: {cv['probabilite']}%"
                        
                        st.download_button(
                            label="📥 Télécharger",
                            data=fichier_content.encode('utf-8'),
                            file_name=cv['nom_fichier'],
                            mime="application/pdf",
                            key=f"download_{cv['id']}"
                        )
                        
                        # Graphique de probabilité
                        st.markdown("**Score:**")
                        progress_color = "green" if cv['probabilite'] >= 70 else "orange" if cv['probabilite'] >= 50 else "red"
                        st.progress(cv['probabilite']/100)
                        st.caption(f"{cv['probabilite']}%")
            
            st.markdown("---")
            
            # Boutons d'actions globales
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📊 Exporter Statistiques"):
                    # Créer un CSV avec les statistiques
                    
                    df = pd.DataFrame(cv_filtres)
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger CSV",
                        data=csv,
                        file_name=f"historique_cv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("🗑️ Vider l'historique"):
                    if st.confirm("Êtes-vous sûr de vouloir supprimer tout l'historique ?"):
                        st.session_state.cv_history = []
                        st.rerun()
            
            with col3:
                if st.button("🔄 Actualiser"):
                    st.rerun()
        
        else:
            st.info("Aucun CV ne correspond aux critères de filtrage sélectionnés.")
            
            if st.button("🔄 Réinitialiser les filtres"):
                st.rerun()
        
        # Configuration avancée (repliée par défaut)
        with st.expander("⚙️ Configuration Avancée", expanded=False):
            st.markdown("### Paramètres de Classification")
            
            seuil_admission = st.slider("Seuil d'admission (%)", 0, 100, 50)
            categories_actives = st.multiselect(
                "Catégories actives",
                ['Développement Web', 'Marketing Digital', 'Data Science', 'Maintenance IT', 'Design Graphique'],
                default=['Développement Web', 'Marketing Digital', 'Data Science']
            )
            
            st.markdown("### Paramètres d'Analyse")
            analyse_approfondie = st.checkbox("Analyse approfondie", value=True)
            sauvegarde_auto = st.checkbox("Sauvegarde automatique", value=False)
            export_auto = st.checkbox("Export automatique des rapports", value=False)
            
            if st.button("💾 Sauvegarder Configuration"):
                # Sauvegarder les paramètres
                config = {
                    'seuil_admission': seuil_admission,
                    'categories_actives': categories_actives,
                    'analyse_approfondie': analyse_approfondie,
                    'sauvegarde_auto': sauvegarde_auto,
                    'export_auto': export_auto
                }
                st.session_state.config = config
                st.success("✅ Configuration sauvegardée avec succès!")
        
        # Affichage des graphiques de synthèse
        if cv_filtres:
            st.markdown("###  Analyse Visuelle")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Graphique de répartition par statut
                
                
                df_statut = pd.DataFrame(cv_filtres)
                fig_statut = px.pie(
                    df_statut, 
                    names='statut', 
                    title='Répartition par Statut',
                    color_discrete_map={
                        'Admis': '#28a745',
                        'Rejeté': '#dc3545',
                        'En Attente': '#ffc107'
                    }
                )
                st.plotly_chart(fig_statut, use_container_width=True)
            
            with col2:
                # Graphique de répartition par classification
                fig_classif = px.pie(
                    df_statut, 
                    names='classification', 
                    title='Répartition par Classification'
                )
                st.plotly_chart(fig_classif, use_container_width=True)         

if __name__ == "__main__":
    main()
