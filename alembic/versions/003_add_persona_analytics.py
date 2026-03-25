"""Add persona analytics tables

Revision ID: 003
Revises: 002
Create Date: 2026-01-11

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '003'
down_revision = '002'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create persona_stats table - Aggregated daily statistics
    op.create_table(
        'persona_stats',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('persona', sa.String(50), nullable=False),
        sa.Column('date', sa.Date(), nullable=False),

        # Usage Metrics
        sa.Column('total_questions', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_conversations', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('unique_sessions', sa.Integer(), nullable=False, server_default='0'),

        # Performance Metrics
        sa.Column('avg_confidence', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('avg_retrieval_time_ms', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('avg_generation_time_ms', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('avg_total_time_ms', sa.Integer(), nullable=False, server_default='0'),

        # Engagement Metrics
        sa.Column('avg_messages_per_conversation', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('avg_session_duration_seconds', sa.Integer(), nullable=False, server_default='0'),

        # Quality Metrics
        sa.Column('helpful_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('unhelpful_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('satisfaction_rate', sa.Float(), nullable=False, server_default='0.0'),

        # Metadata
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),

        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('persona', 'date', name='uq_persona_date')
    )

    # Create indexes for persona_stats
    op.create_index('idx_persona_stats_persona_date', 'persona_stats', ['persona', sa.text('date DESC')])
    op.create_index('idx_persona_stats_date', 'persona_stats', [sa.text('date DESC')])

    # Create message_feedback table - User ratings on responses
    op.create_table(
        'message_feedback',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('message_id', sa.Integer(), nullable=False),
        sa.Column('helpful', sa.Boolean(), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=True),
        sa.Column('comment', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),

        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['message_id'], ['messages.id'], ondelete='CASCADE'),
        sa.UniqueConstraint('message_id', name='uq_message_feedback_message_id'),
        sa.CheckConstraint('rating >= 1 AND rating <= 5', name='ck_rating_range')
    )

    # Create indexes for message_feedback
    op.create_index('idx_message_feedback_message_id_helpful', 'message_feedback', ['message_id', 'helpful'])
    op.create_index('idx_message_feedback_created_at', 'message_feedback', [sa.text('created_at DESC')])
    op.create_index('idx_message_feedback_helpful', 'message_feedback', ['helpful'])

    # Create persona_interactions table - Per-session analytics
    op.create_table(
        'persona_interactions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('conversation_id', sa.Integer(), nullable=False),
        sa.Column('persona', sa.String(50), nullable=False),
        sa.Column('language', sa.String(2), nullable=False),

        # Interaction Counts
        sa.Column('question_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('answer_count', sa.Integer(), nullable=False, server_default='0'),

        # Performance
        sa.Column('avg_confidence', sa.Float(), nullable=False, server_default='0.0'),
        sa.Column('total_response_time_ms', sa.Integer(), nullable=False, server_default='0'),

        # Persona Behavior
        sa.Column('personas_used', sa.JSON(), nullable=True),
        sa.Column('persona_switches', sa.Integer(), nullable=False, server_default='0'),

        # Metadata
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.text('now()')),

        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE')
    )

    # Create indexes for persona_interactions
    op.create_index('idx_persona_interactions_persona_created', 'persona_interactions', ['persona', sa.text('created_at DESC')])
    op.create_index('idx_persona_interactions_conversation_id', 'persona_interactions', ['conversation_id'])
    op.create_index('idx_persona_interactions_created_at', 'persona_interactions', [sa.text('created_at DESC')])


def downgrade() -> None:
    # Drop persona_interactions table
    op.drop_index('idx_persona_interactions_created_at', table_name='persona_interactions')
    op.drop_index('idx_persona_interactions_conversation_id', table_name='persona_interactions')
    op.drop_index('idx_persona_interactions_persona_created', table_name='persona_interactions')
    op.drop_table('persona_interactions')

    # Drop message_feedback table
    op.drop_index('idx_message_feedback_helpful', table_name='message_feedback')
    op.drop_index('idx_message_feedback_created_at', table_name='message_feedback')
    op.drop_index('idx_message_feedback_message_id_helpful', table_name='message_feedback')
    op.drop_table('message_feedback')

    # Drop persona_stats table
    op.drop_index('idx_persona_stats_date', table_name='persona_stats')
    op.drop_index('idx_persona_stats_persona_date', table_name='persona_stats')
    op.drop_table('persona_stats')
