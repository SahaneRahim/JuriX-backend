-- Script SQL pour réinitialiser complètement la base de données JuriX
-- 
-- Ce script:
-- 1. Supprime tous les documents et articles
-- 2. Supprime toutes les conversations RAG  
-- 3. Réinitialise les séquences (IDs recommencent à 1)
--
-- ⚠️  ATTENTION: Cette opération est IRRÉVERSIBLE!
--
-- Usage: psql -d jurix_db -f scripts/reset_database.sql

BEGIN;

-- 1. Supprimer tous les articles (cascade supprimera les embeddings)
DELETE FROM articles;

-- 2. Supprimer tous les documents
DELETE FROM laws;

-- 3. Supprimer les conversations RAG (si les tables existent)
DELETE FROM chat_messages;
DELETE FROM conversations;

-- 4. Réinitialiser les séquences PostgreSQL (IDs recommencent à 1)
ALTER SEQUENCE laws_id_seq RESTART WITH 1;
ALTER SEQUENCE articles_id_seq RESTART WITH 1;
ALTER SEQUENCE categories_id_seq RESTART WITH 1;
ALTER SEQUENCE conversations_id_seq RESTART WITH 1;
ALTER SEQUENCE chat_messages_id_seq RESTART WITH 1;

COMMIT;

-- Vérification
SELECT 
    'laws' as table_name, 
    COUNT(*) as count 
FROM laws
UNION ALL
SELECT 
    'articles', 
    COUNT(*) 
FROM articles
UNION ALL
SELECT 
    'conversations', 
    COUNT(*) 
FROM conversations;
