import pandas as pd
import pytest
from src.fetch_candles import fetch_candles

def test_fetch_candles_signature():
    """
    ทดสอบว่า fetch_candles คืนค่าเป็น DataFrame
    และมีคอลัมน์ที่ต้องการ
    """
    # เรียกฟังก์ชันโดยกำหนด n=1 แท่ง เพื่อหลีกเลี่ยงโหลดข้อมูลเยอะ
    df = fetch_candles(n=1)
    # df อาจเป็น empty DataFrame (ถ้า MT5 ไม่พร้อม) แต่ signature ต้องถูกต้อง
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        # ถ้ามีข้อมูล ให้ตรวจว่ามีคอลัมน์หลักครบ
        for col in ["time", "open", "high", "low", "close", "tick_volume"]:
            assert col in df.columns

@pytest.mark.skip("ต้องการ MT5 เชื่อมต่อจริงจึงรันเทสนี้")
def test_fetch_candles_content():
    """
    ทดสอบเชิงลึก: ถ้า MT5 พร้อม จะต้องคืน DataFrame ที่มีแถวอย่างน้อย 1 แท่ง
    """
    df = fetch_candles(n=5)
    assert not df.empty
    assert len(df) >= 1
