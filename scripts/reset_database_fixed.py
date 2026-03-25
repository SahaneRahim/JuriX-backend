"""
Script SIMPLE et ROBUSTE pour vider la base de données.
Utilise la méthode la plus directe avec gestion d'erreurs complète.
"""

import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

# Configuration
DATABASE_URL = "postgresql://jurix:jurix_dev_password_change_in_prod@localhost:5432/jurix_db"

def reset_database_fixed():
    """Vide la base de données de manière robuste."""
    
    print("=" * 70)
    print("🚀 RÉINITIALISATION DE LA BASE DE DONNÉES")
    print("=" * 70)
    print()
    
    try:
        # Connexion à PostgreSQL avec autocommit
        print("📡 Connexion à PostgreSQL...")
        conn = psycopg2.connect(DATABASE_URL)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)  # Chaque commande est immédiate
        cursor = conn.cursor()
        print("   ✅ Connecté (mode AUTOCOMMIT)")
        print()
        
        # 1. Compter avant suppression
        cursor.execute("SELECT COUNT(*) FROM laws")
        laws_before = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM articles")
        articles_before = cursor.fetchone()[0]
        
        print(f"📊 État AVANT:")
        print(f"   • Documents: {laws_before}")
        print(f"   • Articles: {articles_before}")
        print()
        
        # 2. TRUNCATE avec CASCADE (le plus efficace)
        print("🗑️  Suppression avec TRUNCATE CASCADE...")
        
        try:
            # Essayer avec toutes les tables d'un coup
            cursor.execute("""
                TRUNCATE TABLE 
                    articles, 
                    laws, 
                    chat_messages, 
                    conversations 
                RESTART IDENTITY CASCADE
            """)
            print("   ✅ Toutes les tables vidées")
            
        except psycopg2.ProgrammingError as e:
            # Si certaines tables n'existent pas, essayer une par une
            print(f"   ⚠️  Erreur TRUNCATE global, essai individuel...")
            
            tables = ['articles', 'chat_messages', 'conversations', 'laws']
            for table in tables:
                try:
                    cursor.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
                    print(f"   ✅ Table '{table}' vidée")
                except Exception as e:
                    print(f"   ⏭️  Table '{table}' ignorée: {str(e)[:40]}")
        
        print()
        
        # 3. Vérifier suppression
        cursor.execute("SELECT COUNT(*) FROM laws")
        laws_after = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM articles")
        articles_after = cursor.fetchone()[0]
        
        print(f"📊 État APRÈS:")
        print(f"   • Documents: {laws_after}")
        print(f"   • Articles: {articles_after}")
        print()
        
        # 4. Réinitialiser séquences (si TRUNCATE n'a pas fonctionné)
        if laws_after > 0 or articles_after > 0:
            print("⚠️  Suppression incomplète, utilisation de DELETE...")
            
            cursor.execute("DELETE FROM articles CASCADE")
            cursor.execute("DELETE FROM laws CASCADE")
            
            cursor.execute("SELECT COUNT(*) FROM laws")
            laws_final = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM articles")
            articles_final = cursor.fetchone()[0]
            
            print(f"   • Documents restants: {laws_final}")
            print(f"   • Articles restants: {articles_final}")
            print()
        
        # 5. Réinitialiser séquences explicitement
        print("🔢 Réinitialisation des séquences...")
        
        sequences = [
            'laws_id_seq',
            'articles_id_seq',
            'categories_id_seq',
            'conversations_id_seq',
            'chat_messages_id_seq'
        ]
        
        for seq in sequences:
            try:
                cursor.execute(f"ALTER SEQUENCE {seq} RESTART WITH 1")
                print(f"   ✅ {seq}")
            except Exception as e:
                print(f"   ⏭️  {seq}: {str(e)[:40]}")
        
        print()
        print("=" * 70)
        print("✅ RÉINITIALISATION TERMINÉE!")
        print("=" * 70)
        print()
        print(f"📊 Bilan:")
        print(f"   • Documents supprimés: {laws_before}")
        print(f"   • Articles supprimés: {articles_before}")
        print(f"   • Documents restants: {laws_after}")
        print(f"   • Articles restants: {articles_after}")
        print()
        print("🚀 Prochains IDs: laws.id=1, articles.id=1")
        print("📤 Prêt pour nouveaux uploads!")
        print()
        
        cursor.close()
        conn.close()
        
    except psycopg2.OperationalError as e:
        print()
        print(f"❌ Erreur de connexion: {e}")
        print()
        print("💡 Solutions:")
        print("   1. Vérifier: docker-compose up -d postgres")
        print("   2. Vérifier le port: netstat -an | findstr :5432")
        print("   3. Tester: docker exec -it jurix_postgres psql -U jurix")
        print()
        
    except Exception as e:
        print()
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        print()


if __name__ == "__main__":
    reset_database_fixed()
