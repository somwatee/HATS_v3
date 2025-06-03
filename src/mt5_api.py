import MetaTrader5 as mt5
from pathlib import Path
import time

class MT5Wrapper:
    def __init__(self, cfg_mt5: dict):
        """
        cfg_mt5 ควรประกอบด้วย:
          {
            "terminal_path": str,
            "server": str,
            "login": int,
            "password": str,
            "timeout": int
          }
        """
        self.cfg = cfg_mt5
        self.initialized = False
        self.initialize_mt5()

    def initialize_mt5(self) -> bool:
        """
        พยายามเชื่อมต่อกับ MT5 Terminal
        คืน True ถ้าสำเร็จ, False ถ้าไม่สำเร็จ
        """
        if self.initialized:
            return True

        if mt5.initialize(
            path=self.cfg["terminal_path"],
            login=self.cfg["login"],
            server=self.cfg["server"],
            password=self.cfg["password"],
            timeout=self.cfg["timeout"],
        ):
            self.initialized = True
            return True
        else:
            print("MT5 initialization failed")
            return False

    def shutdown(self):
        """
        ปิดการเชื่อมต่อ MT5 Terminal
        """
        if self.initialized:
            mt5.shutdown()
            self.initialized = False

    def open_order(self, symbol: str, side: str, lot: float = 0.01, sl: float = None, tp: float = None) -> bool:
        """
        เปิดออร์เดอร์ Market Order
        Args:
          symbol: ตัวอย่าง "XAUUSD"
          side: "BUY" หรือ "SELL"
          lot: ขนาดล็อต
          sl: ระบุ Stop Loss (ถ้าไม่ต้องการกำหนด ให้เป็น None)
          tp: ระบุ Take Profit (ถ้าไม่ต้องการกำหนด ให้เป็น None)
        คืน True ถ้าสั่งคำสั่งสำเร็จ, False ถ้าไม่สำเร็จ
        """
        if not self.initialized:
            if not self.initialize_mt5():
                return False

        # ตรวจดูว่า symbol ถูกเปิดใช้งานใน MT5 หรือไม่
        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"Symbol {symbol} not found")
            return False
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                print(f"Failed to select symbol {symbol}")
                return False

        # ดึงราคา Bid/Ask ปัจจุบัน
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print("Failed to get price tick")
            return False

        price = tick.ask if side.upper() == "BUY" else tick.bid
        order_type = mt5.ORDER_TYPE_BUY if side.upper() == "BUY" else mt5.ORDER_TYPE_SELL

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "deviation": 10,
            "magic": 123456,
            "comment": "Hybrid AI EA",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        # ถ้ามี SL/TP ให้ใส่ใน request
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp

        result = mt5.order_send(request)
        if result is None:
            print("Order send returned None")
            return False
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed, retcode={result.retcode}")
            return False
        return True

    def close_all(self, symbol: str) -> bool:
        """
        ปิดทุกตำแหน่งที่เปิดค้างอยู่ของ symbol นั้น ๆ ด้วย Market Order
        คืน True อย่างน้อยปิดได้หนึ่งตำแหน่ง, False ถ้าไม่มีตำแหน่งหรือปิดล้มเหลว
        """
        if not self.initialized:
            if not self.initialize_mt5():
                return False

        positions = mt5.positions_get(symbol=symbol)
        if positions is None or len(positions) == 0:
            # ไม่มีตำแหน่งเปิดค้าง
            return False

        success_any = False
        for pos in positions:
            ticket = pos.ticket
            volume = pos.volume
            side = "SELL" if pos.type == mt5.POSITION_TYPE_BUY else "BUY"  # ปิด BUY ต้อง SELL กลับ
            price = mt5.symbol_info_tick(symbol).bid if side == "SELL" else mt5.symbol_info_tick(symbol).ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": ticket,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_SELL if side == "SELL" else mt5.ORDER_TYPE_BUY,
                "price": price,
                "deviation": 10,
                "magic": 123456,
                "comment": "Hybrid AI EA close",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                success_any = True
            else:
                print(f"Failed to close position {ticket}, retcode={getattr(result, 'retcode', 'N/A')}")
        return success_any
