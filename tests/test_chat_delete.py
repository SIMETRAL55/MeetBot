import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from meetbot.db.models import ChatMessage, ChatSession, Job, User, Base
from meetbot.web.api import api_delete_chat_message
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, StaticPool
from sqlalchemy.orm import sessionmaker

# Use a shared in-memory SQLite for testing logic
engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)

@pytest.mark.asyncio
async def test_api_delete_chat_message_pair(db_session):
    """Test deleting a user message and its following assistant response."""
    user_id = "user-123"
    
    user = User(id=user_id, username="testuser", password_hash="hash")
    db_session.add(user)
    job = Job(id="job-1", user_id=user_id, filename="test.mp3", original_filename="test.mp3")
    db_session.add(job)
    db_session.commit()
    
    session = ChatSession(id="sess-1", job_id=job.id)
    db_session.add(session)
    db_session.commit()
    
    msg_u = ChatMessage(id="msg-u-1", role="user", session_id=session.id, content="Q", created_at=datetime.now(timezone.utc))
    db_session.add(msg_u)
    db_session.commit()
    
    msg_a = ChatMessage(id="msg-a-1", role="assistant", session_id=session.id, content="A", created_at=datetime.now(timezone.utc) + timedelta(seconds=1))
    db_session.add(msg_a)
    db_session.commit()

    request = MagicMock()
    # Patch WHERE it is defined, so local imports get the patch
    with patch("meetbot.web.auth_middleware.get_optional_user", return_value={"sub": user_id}), \
         patch("meetbot.db.database.get_session", return_value=TestingSessionLocal):
        
        response = await api_delete_chat_message(request, "msg-u-1")

    assert response.status_code == 204
    
    # Refresh/New session to verify deletion
    db2 = TestingSessionLocal()
    assert db2.query(ChatMessage).filter(ChatMessage.id == "msg-u-1").first() is None
    assert db2.query(ChatMessage).filter(ChatMessage.id == "msg-a-1").first() is None
    db2.close()

@pytest.mark.asyncio
async def test_api_delete_orphan_user_message(db_session):
    """Test deleting a user message that has no assistant response."""
    user_id = "user-456"
    
    user = User(id=user_id, username="testuser2", password_hash="hash")
    db_session.add(user)
    job = Job(id="job-2", user_id=user_id, filename="test.mp3", original_filename="test.mp3")
    db_session.add(job)
    db_session.commit()
    
    session = ChatSession(id="sess-2", job_id=job.id)
    db_session.add(session)
    db_session.commit()
    
    msg_u = ChatMessage(id="msg-u-orphan", role="user", session_id=session.id, content="Q alone", created_at=datetime.now(timezone.utc))
    db_session.add(msg_u)
    db_session.commit()

    request = MagicMock()
    with patch("meetbot.web.auth_middleware.get_optional_user", return_value={"sub": user_id}), \
         patch("meetbot.db.database.get_session", return_value=TestingSessionLocal):
        
        response = await api_delete_chat_message(request, "msg-u-orphan")

    assert response.status_code == 204
    db2 = TestingSessionLocal()
    assert db2.query(ChatMessage).filter(ChatMessage.id == "msg-u-orphan").first() is None
    db2.close()

@pytest.mark.asyncio
async def test_api_delete_unauthorized(db_session):
    """Test that a user cannot delete someone else's message."""
    user_id_a = "user-A"
    user_id_b = "user-B"
    
    user_a = User(id=user_id_a, username="usera", password_hash="hash")
    user_b = User(id=user_id_b, username="userb", password_hash="hash")
    db_session.add(user_a)
    db_session.add(user_b)
    db_session.commit()
    
    job_a = Job(id="job-a", user_id=user_id_a, filename="test.mp3", original_filename="test.mp3")
    db_session.add(job_a)
    db_session.commit()
    
    session_a = ChatSession(id="sess-a", job_id=job_a.id)
    db_session.add(session_a)
    db_session.commit()
    
    msg_u = ChatMessage(id="msg-u-a", role="user", session_id=session_a.id, content="Q from A", created_at=datetime.now(timezone.utc))
    db_session.add(msg_u)
    db_session.commit()

    request = MagicMock()
    with patch("meetbot.web.auth_middleware.get_optional_user", return_value={"sub": user_id_b}), \
         patch("meetbot.db.database.get_session", return_value=TestingSessionLocal):
        
        with pytest.raises(HTTPException) as exc_info:
            await api_delete_chat_message(request, "msg-u-a")

    assert exc_info.value.status_code == 404
    db2 = TestingSessionLocal()
    assert db2.query(ChatMessage).filter(ChatMessage.id == "msg-u-a").first() is not None
    db2.close()
