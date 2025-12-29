#!/usr/bin/env python3
"""
Test rapide de predict.py corrigé

Vérifie que :
1. Le bon modèle est chargé
2. Les cutoff_date fonctionnent
3. Les distribution shift warnings s'affichent
"""

import sys
from pathlib import Path
from datetime import datetime

# Ajouter le chemin vers predict_CORRECTED
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("TEST DE predict.py CORRIGÉ")
print("=" * 70)

# Importer le module corrigé
try:
    # Si tu as déjà remplacé le fichier
    from venture_scope.ml.predict import predict_startup, load_model
    print("✅ Import depuis venture_scope.ml.predict")
except ImportError:
    # Sinon, importer depuis le fichier corrigé
    import importlib.util
    spec = importlib.util.spec_from_file_location("predict", "predict_CORRECTED.py")
    predict = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(predict)
    predict_startup = predict.predict_startup
    load_model = predict.load_model
    print("✅ Import depuis predict_CORRECTED.py")

print("\n" + "=" * 70)
print("TEST 1: Vérifier que le modèle temporal est chargé par défaut")
print("=" * 70)

try:
    model = load_model(use_temporal=True)
    if model:
        print("✅ PASS: Modèle temporal chargé avec succès")
        if hasattr(model, 'n_estimators'):
            print(f"   ℹ {model.n_estimators} arbres détectés")
    else:
        print("❌ FAIL: Modèle non trouvé")
        print("   ℹ Exécute d'abord: python src/venture_scope/ml/model_temporal.py")
except Exception as e:
    print(f"❌ FAIL: Erreur lors du chargement: {e}")

print("\n" + "=" * 70)
print("TEST 2: Prédiction simple (sans cutoff_date)")
print("=" * 70)

try:
    result = predict_startup(
        funding_amount=10_000_000,
        stage='Series A',
        sector='saas',
        country='USA',
        investors_count=5,
        founded_year=2020
    )
    
    if result:
        print("✅ PASS: Prédiction réussie")
        print(f"   ℹ Success probability: {result['success_probability']*100:.1f}%")
        print(f"   ℹ Investment score: {result['kpis']['investment_score']:.1f}/100")
        
        # Vérifier warning shift
        if result['shift_check']['has_shift']:
            print("   ✅ Distribution shift warning détecté (attendu pour 2025)")
        else:
            print("   ⚠ Pas de warning shift (inattendu)")
    else:
        print("❌ FAIL: Prédiction a échoué")
        
except Exception as e:
    print(f"❌ FAIL: Erreur: {e}")

print("\n" + "=" * 70)
print("TEST 3: Prédiction temporelle (avec cutoff_date 2011)")
print("=" * 70)

try:
    result = predict_startup(
        funding_amount=5_000_000,
        stage='Series A',
        sector='saas',
        country='USA',
        investors_count=4,
        founded_year=2008,
        cutoff_date=datetime(2011, 12, 31)
    )
    
    if result:
        print("✅ PASS: Prédiction temporelle réussie")
        print(f"   ℹ Success probability: {result['success_probability']*100:.1f}%")
        
        # Vérifier PAS de warning shift
        if not result['shift_check']['has_shift']:
            print("   ✅ Pas de warning shift (attendu pour 2011)")
        else:
            print("   ⚠ Warning shift inattendu pour 2011")
            
        # Vérifier KPI calculés avec cutoff_date
        if result['kpis'].get('cutoff_date'):
            print(f"   ✅ Cutoff date utilisé: {result['kpis']['cutoff_date'].date()}")
        else:
            print("   ⚠ Cutoff date non enregistré dans KPIs")
    else:
        print("❌ FAIL: Prédiction temporelle a échoué")
        
except Exception as e:
    print(f"❌ FAIL: Erreur: {e}")

print("\n" + "=" * 70)
print("TEST 4: Vérifier feature contributions")
print("=" * 70)

try:
    result = predict_startup(
        funding_amount=15_000_000,
        stage='Series B',
        sector='fintech',
        country='USA',
        investors_count=8,
        founded_year=2018
    )
    
    if result and result.get('feature_contributions'):
        print("✅ PASS: Feature contributions disponibles")
        print("   Top 3 features:")
        for i, (feat, imp) in enumerate(list(result['feature_contributions'].items())[:3]):
            print(f"     {i+1}. {feat}: {imp*100:.1f}%")
    else:
        print("⚠ WARNING: Feature contributions non disponibles")
        print("   (Normal si modèle n'a pas feature_importances_)")
        
except Exception as e:
    print(f"❌ FAIL: Erreur: {e}")

print("\n" + "=" * 70)
print("RÉSUMÉ DES TESTS")
print("=" * 70)
print("""
✅ Si tous les tests PASS :
   → predict.py est correctement corrigé
   → Tu peux l'utiliser en production
   → Remplace l'ancien fichier si ce n'est pas déjà fait

⚠ Si certains tests échouent :
   → Vérifie que random_forest_temporal.pkl existe
   → Vérifie les paths (data/models/ vs results/models/)
   → Consulte CORRECTIONS_PREDICT_SUMMARY.md

📝 Prochaines étapes :
   1. Backup l'ancien: cp predict.py predict_OLD.py
   2. Remplace: cp predict_CORRECTED.py predict.py
   3. Test en interactive: python src/venture_scope/ml/predict.py
""")

print("=" * 70)
print("TEST TERMINÉ")
print("=" * 70)