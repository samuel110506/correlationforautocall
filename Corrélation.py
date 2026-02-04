import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import streamlit as st
import matplotlib.colors as mcolors
import seaborn as sns
import itertools

def récupération():
    df = pd.read_excel("Liste stocks.xlsx")
    df['Nom du stock'] = df["Nom de l'entreprise"] + " (" + df['Ticker Yahoo Finance'] + ")"
    return df
    
tickers=récupération()
st.title("Sélection des sous-jacents")
options_affichage = tickers['Nom du stock'].tolist()
n_assets = st.sidebar.number_input("Taille du panier (ex: 3 actions)", min_value=2, max_value=5)
selection=st.multiselect("Choisissez vos actions (max 10) :",options=options_affichage,max_selections=10)
selection = tickers[tickers['Nom du stock'].isin(selection)]['Ticker Yahoo Finance'].tolist()
min_corr_threshold = 0.4
period_choice = st.sidebar.selectbox(
    "Période d'analyse historique :",
    options=["1y", "2y", "5y", "10y"],
    index=2  # Par défaut sur "5y"
)


if len(selection) > 0:
    data = yf.download(selection, period=period_choice,auto_adjust=True)['Close']
    returns = np.log(data / data.shift(1)).dropna()
    corr_matrix = returns.corr()
    colors = ["#ff4c4c", "#ffff8d", "#4caf50"] # Rouge, Jaune, Vert, Jaune
    nodes = [-1.0, 0.4, 0.7, 1.0]

    # Création de la colormap personnalisée
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(nodes, cmap.N)
    
    # 2. Affichage de la Heatmap
    st.subheader("Matrice de Corrélation Spéciale Autocall")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap=cmap, 
        norm=norm,    # On applique les limites définies
        center=None, 
        fmt=".2f", 
        linewidths=0.5,
        ax=ax
    )
    
    # Ajout d'une légende textuelle pour plus de clarté
    st.pyplot(fig)
    
    st.markdown("""
    **Légende métier :**
    - 🔴 **Rouge (< 0.4)** : Corrélation faible voire négative donc trop risqué pour un autocall.
    - 🟡 **Jaune (0.4 à 0.7)** : Zone optimale pour le couple rendement/risque.
    - 🟢 **Vert (> 0.7)** : Corrélation très forte. Rendement (coupon) potentiellement faible.
    """)
    all_combos = list(itertools.combinations(selection, n_assets))
    valid_baskets = []
    for combo in all_combos:
            sub_corr = corr_matrix.loc[list(combo), list(combo)]
            
            # On extrait les valeurs de corrélation (hors diagonale de 1.0)
            # On vérifie si le minimum de corrélation dans le panier est > 0.4
            mask = ~np.eye(sub_corr.shape[0], dtype=bool)
            min_corr_in_basket = sub_corr.values[mask].min()
            avg_corr_in_basket = sub_corr.values[mask].mean()

            if min_corr_in_basket >= min_corr_threshold:
                valid_baskets.append({
                    "Panier": combo,
                    "Correl Min": round(min_corr_in_basket, 2),
                    "Correl Moy": round(avg_corr_in_basket, 2)
                })
    # 4. Affichage sous forme de tableau interactif
    if valid_baskets:
        st.success(f"### {len(valid_baskets)} Paniers Éligibles trouvés")
        
        # Création du DataFrame
        df_res = pd.DataFrame(valid_baskets)
        
        # On trie par Corrélation Moyenne pour mettre les meilleurs coupons en haut (les plus proches de 0.4)
        df_res = df_res.sort_values("Correl Moy", ascending=True)
    
        # Affichage stylisé
        st.dataframe(
            df_res,
            column_config={
                "Panier": st.column_config.TextColumn("Composition du Panier"),
                "Correl Min": st.column_config.NumberColumn("Correl Min 📉", format="%.2f"),
                "Correl Moy": st.column_config.ProgressColumn("Correl Moy 📊", min_value=0.4, max_value=1.0, format="%.2f")
            },
            use_container_width=True,
            hide_index=True
        )
        
        st.info("**Astuce Coupon** : Les paniers en haut de liste (barre de progression courte) sont ceux qui offrent potentiellement le rendement le plus élevé. Les paniers en bas de liste (barre de progression longue) sont les moins risqués.")
    else:
        st.warning("Aucun panier ne respecte le critère de 0.4 minimum entre chaque actif.")
                         

