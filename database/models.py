from sqlalchemy import Column, Integer, String, Float, TIMESTAMP
from datetime import datetime
from database.db import Base

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    description = Column(String, index=True)
    amount = Column(Float)
    category = Column(String)
    predicted_category = Column(String)
    user_corrected = Column(Integer, default=0)
    created_at = Column(TIMESTAMP, default=datetime.utcnow)

    def __repr__(self):
        return f"<Transaction(description='{self.description}', category='{self.category}')>" 