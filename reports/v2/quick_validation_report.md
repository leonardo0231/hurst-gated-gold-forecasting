# V2 Quick Validation Report

## وضعیت اجرا

- تاریخ بررسی: 2026-08-06
- محیط: Python 3.13.5
- نوع داده: deterministic synthetic research fixture
- هدف: بررسی صحت نرم‌افزار، نه ارزیابی بازار واقعی

## کنترل کیفیت

- `pytest`: **7 passed**
- Coverage کل package V2: **88%**
- `compileall`: **PASS**
- `ruff`: اجرا نشد؛ executable در محیط موجود نبود
- `mypy`: اجرا نشد؛ executable در محیط موجود نبود
- اجرای heavy profile: در محدودیت زمان محیط کامل نشد
- اجرای quick-validation چهار افق: **PASS**

## نتایج quick-validation

| افق | Strategy | Balanced Accuracy | Macro-F1 | وضعیت gate |
|---:|---|---:|---:|---|
| 1 | Random Forest | 0.728 | 0.728 | PASS |
| 5 | Random Forest | 0.779 | 0.779 | PASS |
| 10 | Extra Trees | 0.814 | 0.810 | PASS |
| 20 | Learned Regime Gate | 0.412 | 0.379 | FAIL |

این نتایج نشان می‌دهند pipeline می‌تواند سیگنال ثبت‌شده fixture را در افق‌های 1، 5 و 10 روزه یاد بگیرد، اما در افق 20 روزه شکست می‌خورد. شکست افق 20 عمداً مخفی یا با تغییر Locked Test اصلاح نشده است.

## نتیجه قابل ادعا

- پیاده‌سازی و تست نرم‌افزار: انجام‌شده
- توان عبور از gate 60% روی fixture: برای افق‌های 1، 5 و 10 تأییدشده
- عبور از 60% روی داده واقعی طلا: بررسی‌نشده و نیازمند dataset معتبر
- سودآوری معاملاتی: اثبات‌نشده
