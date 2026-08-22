# ممیزی مستقل نهایی اصلاحات QA ـ خانواده v2

تاریخ ممیزی: 2026-08-22  
دامنه: worktree نهایی شاخه `thesis-v2-rebuild` و اجرای immutable
`executable_direction_hurst_ablation_v2-20260822T165905Z`  
توصیه انتشار: **`reject`**

این رأی به معنی شکست کنترل‌های مهندسی نیست؛ به این معنی است که هیچ مدل یا ادعای عملکردی/اقتصادی از شواهد فعلی قابل انتشار نیست. هر ۱۲ آزمایش v2 رد شده‌اند، هیچ candidate فریز نشده و هیچ confirmation، audit کاندید یا MT5 candidate verification اجرا نشده است.

## یافته‌های مستقل

| اولویت | یافته و شاهد | فایل‌های متأثر | ریسک | اقدام پیشنهادی | وضعیت |
|---|---|---|---|---|---|
| P0 | در هر ۶۰ انتخاب outer، مرز ثانویه calibration/evaluation با `executable_label_end_index` purge نشده است؛ sigmoid در ۲۶ انتخاب برگزیده شد. | `src/hge_gold/research_experiments_v2.py`، inner-selection artifacts، registry و metrics v2 | metrics برای promotion یا ادعای leakage-free معتبر نیستند. | v2 را تغییر ندهید؛ در یک خانواده پیش‌ثبت‌شده تازه این مرز را purge و تست منفی اضافه کنید. | deferred؛ خانواده تازه لازم است |
| P0 | registry تعریف fold اصلی را `inner_purged_walk_forward_only` ثبت می‌کند، اما defect مرز ثانویه پس از freeze در خود run receipt ثبت نشده است. عبارت صریح `leakage_free=true` در registry/receipt یافت نشد، ولی metadata frozen نیز این defect پسینی را منعکس نمی‌کند. | run receipt، registry، QA reports | خواننده ممکن است «inner purged» را به‌اشتباه به کل calibration selection تعمیم دهد. | این گزارش مستقل و گزارش canonical را به‌عنوان superseding QA کنار run حفظ و hash/commit کنید. | با این گزارش پیاده‌سازی شد؛ run immutable ماند |
| P0 | ۱۲/۱۲ trial رد، candidate directory غایب، و confirmation/audit/MT5 candidate run اجرا نشده است. | experiment inventory، promotion decisions، candidate artifacts | هیچ مدل قابل انتشار، توصیه معامله یا ادعای سودآوری وجود ندارد. | release مدل را رد و نتیجه منفی را حفظ کنید. | implemented |
| P1 | paired block CI برای Hurst در برابر no-Hurst وجود ندارد؛ common-calendar دقیق، familywise CI و event-boundary-safe PBO/DSR نیز کامل نیستند. | statistical artifacts و deferred register | ادعای ارزش افزوده Hurst یا اتکا به PBO/DSR نامعتبر است. | فقط در خانواده جدید و با بودجه مستقل اصلاح شود. | deferred |
| P1 | `freeze_candidate()` از overwrite/hash mutation جلوگیری می‌کند، اما eligibility را از registry/finalization به‌طور مستقل resolve نمی‌کند و به boolean caller اعتماد دارد. | `src/hge_gold/research_protocol.py` | freeze صوری در فراخوانی آینده ممکن است؛ در این run رخ نداده است. | پیش از هر candidate آینده، registry decision، receipt، artifact hashes، gates و QA approval داخل API verify شوند. | deferred قبل از candidate آینده |
| مثبت | جداسازی فیزیکی audit، hash-chain registry، receipt شامل ۱۱۰ خروجی و ۲۱ source، quarantine قابل بازیابی، اقتصاد non-overlapping و قرارداد محدود parity معتبرند. | partitions، registry/receipt، quarantine، `execution_v2.py` | این نقاط به‌تنهایی اعتبار مدل ایجاد نمی‌کنند. | به‌عنوان زیرساخت حفظ شوند. | implemented |

## کنترل شواهد

- development: ۳۲۲۱ ردیف، 2011-01-03 تا 2023-06-30؛ قبلاً دیده‌شده.
- historical audit: ۷۹۶ ردیف، 2023-07-03 تا 2026-07-31؛ قبلاً آشکارشده و در v2 بارگذاری نشده است.
- receipt: ۱۱۰ خروجی و ۲۱ منبع runtime؛ validation مجدد موفق.
- registry: ۱۲ رکورد reject با head برابر `183fb0604debc7eca7791b96d1de331e9ca24acb50d1903a963dda0f7d4074b6`.
- بهترین نقطه توسعه: no-Hurst/H1 با BA برابر 0.520347 و macro-F1 برابر 0.516177؛ بازده خالص ۵ و ۱۰ bps منفی است.
- شواهد confirmation تازه: وجود ندارد.
- شواهد future out-of-sample: وجود ندارد.

## نتیجه انتشار

**`reject`** برای انتشار مدل، ادعای promotion، سودآوری، برتری Hurst، parity بومی MQL5 یا عملکرد آینده. انتشار کد و آرتیفکت‌ها صرفاً به‌عنوان یک خروجی پژوهشی منفی، بازتولیدپذیر و دارای محدودیت‌های صریح مجاز است.

## Skills Used

عامل مستقل از ممیزی کد/آرتیفکت فقط‌خواندنی استفاده کرد. هیچ skill یا plugin نامرتبط GitHub، Hugging Face، Jupyter یا Vercel در این ممیزی به‌کار نرفت.
