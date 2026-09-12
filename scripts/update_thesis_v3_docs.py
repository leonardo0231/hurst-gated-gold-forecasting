"""Update the revised Persian thesis with the finalized Hurst-ablation v3 results.

The script intentionally edits only the already revised thesis.  All numerical
values are read from the immutable v3 run artifacts rather than retyped from a
previous report.
"""

# The paragraph literals intentionally preserve long Persian prose chunks.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from docx import Document
from PIL import Image, ImageDraw

RUN_ID = "executable_direction_hurst_ablation_v3-20260912T082354Z"
DEFAULT_DOC = Path("docs/پایان_نامه_HGE_Gold_Forecasting_بازنگری‌شده.docx")
DEFAULT_RUN = Path("artifacts/research/runs") / RUN_ID


def _replace_paragraph(paragraph: Any, text: str) -> None:
    """Replace paragraph text while retaining the first run's formatting."""

    properties = (
        deepcopy(paragraph.runs[0]._r.rPr)
        if paragraph.runs and paragraph.runs[0]._r.rPr is not None
        else None
    )
    for child in list(paragraph._p):
        if child.tag.endswith("}r") or child.tag.endswith("}hyperlink"):
            paragraph._p.remove(child)
    run = paragraph.add_run(text)
    if properties is not None:
        run._r.insert(0, properties)


def _replace_cell(cell: Any, text: str) -> None:
    _replace_paragraph(cell.paragraphs[0], text)
    for paragraph in cell.paragraphs[1:]:
        _replace_paragraph(paragraph, "")


def _persian_digits(value: str) -> str:
    return value.translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")).replace(".", "٫")


def _pct(value: float) -> str:
    return f"{_persian_digits(f'{value * 100:.2f}')}٪"


def _signed_pp(value: float) -> str:
    sign = "+" if value >= 0 else "−"
    return f"{sign}{_persian_digits(f'{abs(value) * 100:.2f}')} واحد درصد"


def _decimal(value: float, digits: int = 4) -> str:
    return _persian_digits(f"{value:.{digits}f}")


def _replace_ablation_chart(
    document_path: Path, by_trial: dict[tuple[str, int], dict[str, Any]]
) -> None:
    """Replace the embedded Figure 5-2 raster with the finalized Table 5 values."""

    arms = ("no_hurst", "current_dfa_hurst", "robust_hurst_regime")
    horizons = (1, 5, 10, 20)
    values = [
        [by_trial[arm, horizon]["pooled_balanced_accuracy"] * 100 for horizon in horizons]
        for arm in arms
    ]
    with ZipFile(document_path, "r") as source:
        chart_bytes = source.read("word/media/image3.png")
        entries = [(info, source.read(info.filename)) for info in source.infolist()]

    image = Image.open(BytesIO(chart_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((125, 120, 1115, 480), fill="white")
    plot_left, plot_right = 130, 1110
    plot_top, plot_bottom = 130, 451
    y_min, y_max = 46.0, 52.0

    def y_coord(value: float) -> int:
        return round(plot_top + (y_max - value) * (plot_bottom - plot_top) / (y_max - y_min))

    for value in (52, 50, 48, 46):
        y = y_coord(value)
        draw.line((plot_left, y, plot_right, y), fill=(220, 220, 220), width=2)
    draw.line((plot_left, y_coord(50), plot_right, y_coord(50)), fill=(192, 0, 0), width=4)
    x_coords = [280, 525, 770, 1015]
    colors = [(68, 114, 196), (237, 125, 49), (112, 173, 71)]
    for series, color in zip(values, colors, strict=True):
        points = [(x, y_coord(value)) for x, value in zip(x_coords, series, strict=True)]
        draw.line(points, fill=color, width=5, joint="curve")
        for x, y in points:
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color)

    updated = BytesIO()
    image.save(updated, format="PNG")
    replacement = updated.getvalue()
    temporary = document_path.with_suffix(".v3.tmp.docx")
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as target:
        for info, content in entries:
            target.writestr(
                info, replacement if info.filename == "word/media/image3.png" else content
            )
    os.replace(temporary, document_path)


def _load(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory = json.loads((run_dir / "experiment_inventory.json").read_text(encoding="utf-8"))
    paired = json.loads((run_dir / "paired_inference.json").read_text(encoding="utf-8"))
    return inventory, paired


def _index_inventory(inventory: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(row["arm"], int(row["horizon"])): row for row in inventory["experiments"]}


def _index_paired(paired: dict[str, Any], key: str) -> dict[int, dict[str, Any]]:
    return {int(row["horizon"]): row for row in paired[key]}


def update_document(document_path: Path, run_dir: Path) -> None:
    inventory, paired = _load(run_dir)
    by_trial = _index_inventory(inventory)
    primary = _index_paired(paired, "primary")

    document = Document(document_path)
    if len(document.paragraphs) < 341 or len(document.tables) < 16:
        raise RuntimeError("The thesis structure changed; refusing index-based edits")

    # Abstract: remove the historical v2 interpretation and state the finalized
    # v3 evidence and its bounded development-only scope.
    paragraph_updates = {
        23: (
            "این پژوهش ارزش افزودهٔ ویژگی‌های رژیمی مبتنی بر نمای هرست را در پیش‌بینی "
            "چندافقی جهت روزانهٔ XAUUSD بررسی می‌کند. داده شامل ۴۰۱۷ مشاهدهٔ روزانه از "
            "نماد XAUUSD در بازهٔ ۲۰۱۱-۰۱-۰۳ تا ۲۰۲۶-۰۷-۳۱ است؛ دادهٔ کارگزار MetaQuotes-Demo، "
            "حجم تیک و بخشی از فرادادهٔ بازسازی‌شده دارد و به‌عنوان آزمون آیندهٔ بکر تلقی نمی‌شود."
        ),
        24: (
            "تصمیم در پایان روز t گرفته و ورود اجرایی در بازشدن روز t+1 تعریف می‌شود. "
            "خانوادهٔ پیش‌ثبت‌شدهٔ v3 شامل سه بازو و چهار افق، یعنی ۱۲ آزمایش است. "
            "اعتبارسنجی گام‌به‌گام، حذف بازهٔ برچسب‌های هم‌پوشان، purge مستقل مرز "
            "کالیبراسیون ثانویه و تقویم مشترک برای مقایسهٔ زوجی به‌کار رفته است."
        ),
        25: (
            "در بازاجرای کامل v3، دقت متوازن تجمیعی بازوی بدون هرست در افق‌های ۱، ۵، ۱۰ و ۲۰ "
            f"به‌ترتیب {_pct(by_trial['no_hurst', 1]['pooled_balanced_accuracy'])}، "
            f"{_pct(by_trial['no_hurst', 5]['pooled_balanced_accuracy'])}، "
            f"{_pct(by_trial['no_hurst', 10]['pooled_balanced_accuracy'])} و "
            f"{_pct(by_trial['no_hurst', 20]['pooled_balanced_accuracy'])} بود. در مقایسهٔ زوجی "
            "اولیه، delta بازوی DFA1 نسبت به بدون هرست در هر چهار افق منفی، همهٔ بازه‌های "
            "۹۵٪ شامل صفر و همهٔ pهای Holm برابر 0.449955 بودند. هر ۱۲ تصمیم رد شد و "
            "هر ۶۰ انتخاب outer پس از purge بدون overlap ثبت شد."
        ),
        26: (
            "نتیجهٔ اصلی محدود و منفی است: در این داده و پروتکل، شواهدی برای ارزش افزودهٔ "
            "پایدارِ ویژگی‌های فعلی DFA1 یا رژیم مقاوم هرست نسبت به بازوی بدون هرست به‌دست نیامد. "
            "این نتیجه، به‌دلیل نقش توسعه‌ای داده و پرچم provenance_complete=false، ادعای "
            "تعمیم به بازار یا قابلیت معامله نیست؛ ارزش پژوهش در اصلاح قابل ممیزی و گزارش صادقانهٔ نتیجه است."
        ),
        152: (
            "در این پروژه برآورد اصلی نمای هرست با DFA1 انجام می‌شود. ورودی، تفاضل مرتبهٔ یکِ "
            "لگاریتم قیمت است؛ پس از مرکززدایی و انتگرال‌گیری تجمعی، برای پنجره‌های ۶۴ و ۱۲۸ "
            "مشاهده، مقیاس‌های توان دو از ۴ تا floor(n_increments/3) ساخته می‌شوند. قطعه‌های "
            "غیرهم‌پوشان از ابتدا و انتهای پنجره با روند خطی مرتبهٔ یک detrend می‌شوند و شیب "
            "رگرسیون log–log نوسان بر مقیاس، برآورد DFA1 است. حداقل ۳۲ مشاهده و حداقل سه مقیاس "
            "قابل استفاده لازم است و انحراف معیار افزایش‌های نزدیک به صفر با کف 1e-12 کنترل می‌شود."
        ),
        190: (
            "مجموعهٔ پایه ۷۰ ویژگی و مجموعهٔ بدون هرست ۶۵ ویژگی دارد. بازوی current_dfa_hurst "
            "ویژگی‌های DFA1 در پنجره‌های ۶۴ و ۱۲۸ و پرچم‌های availability را اضافه می‌کند؛ "
            "بازوی robust_hurst_regime میانه، پراکندگی، آستانه‌های صدکی و regime را اضافه می‌کند. "
            "رژیم مقاوم با پنجرهٔ ۲۵۲، حداقل ۱۲۶ مشاهده، صدک‌های ۰٫۳۳ و ۰٫۶۷ و خلاصهٔ یک‌دوره‌باوقفهٔ "
            "هرست ساخته می‌شود. همهٔ انتقال‌ها و آستانه‌ها فقط از گذشته استفاده می‌کنند."
        ),
        196: (
            "دسته‌آزمایش پیش‌ثبت‌شدهٔ v3 دارای سه بازو و چهار افق، در مجموع ۱۲ آزمایش است: بدون "
            "هرست، بازنمایی جاری DFA1 و رژیم هرست مقاوم. در هر سه بازو pipeline یکسان شامل "
            "SimpleImputer با strategy=median و missing-indicator، StandardScaler و LogisticRegression "
            "متعادل با C=0.2، penalty=l2، solver=lbfgs، max_iter=2000، tol=0.0001 و random_state=42 "
            "است. کالیبراسیون اختیاری sigmoid با حداقل ۴۰ نمونه و C=1000000، و حاشیه‌های بدون معاملهٔ "
            "۰ و ۰٫۰۵ در انتخاب درونی بررسی شدند؛ انتخاب با ۳ inner fold و ارزیابی با ۵ outer fold "
            "انجام شد. پس از پایان ۱۲ آزمایش، تغییر انتخابی برای یافتن نتیجهٔ مطلوب مجاز نیست."
        ),
        212: (
            "کالیبراسیون احتمال پس از تولید پیش‌بینی خارج از آموزش انجام می‌شود. در v3، endpoint "
            "اجرایی هر برچسب به‌صورت مستقل تا مرز کالیبراسیون حمل و قاعدهٔ بستهٔ "
            "calibration_label_end_index < evaluation_start_row_id اعمال شد. این purge در هر ۶۰ "
            "انتخاب outer overlap صفر داشت و sigmoid فقط پس از کنترل eligibility و وجود دو کلاس برازش شد."
        ),
        214: (
            "برای مقایسهٔ زوجی، balanced accuracy با moving-block bootstrap با طول بلوک ۱۰ روز، "
            "۱۰٬۰۰۰ تکرار و سطح اطمینان ۹۵٪ برآورد شد. آزمون دوطرفهٔ sign-flip روی بلوک‌های زمانی "
            "غیرهم‌پوشان نیز ۱۰٬۰۰۰ جایگشت داشت؛ این طراحی وابستگی زمانی را بهتر از بازنمونه‌گیری "
            "مستقل هر مشاهده منعکس می‌کند."
        ),
        216: (
            "در paired inference، row_id و y_true بازوی current_dfa_hurst با no_hurst در هر افق "
            "دقیقاً یکسان بود؛ تعداد ردیف‌های زوجی H1، H5، H10 و H20 به‌ترتیب ۲۳۷۷، ۲۳۶۹، "
            "۲۳۵۹ و ۲۳۳۹ است. برای contrast اصلی DFA1−no-Hurst، deltaهای دقت متوازن به‌ترتیب "
            "−۱٫۴۹، −۰٫۹۳، −۳٫۲۳ و −۲٫۶۲ واحد درصد بود؛ همهٔ CIها صفر را پوشش دادند و pهای "
            "Holm برابر ۰٫۴۴۹۹۵۵ شدند. اصلاح Holm در هر خانوادهٔ چهارافقی جداگانه اعمال شد."
        ),
        283: (
            "جدول ۵-۲ دقت متوازن تجمیعی سه بازو را در اجرای نهایی v3 نشان می‌دهد. بازوی بدون هرست "
            "در H1، H5 و H10 از بازوی DFA1 بالاتر است و بازوی مقاوم فقط در H5 و H20 از بدون هرست "
            "بالاتر قرار گرفت. مقایسهٔ زوجی اولیه در هیچ افقی برتری معنادار DFA1 را نشان نداد؛ "
            "بنابراین جابه‌جایی بازوی برتر به‌تنهایی شاهد ارزش افزودهٔ پایدار نیست."
        ),
        285: (
            "منبع: experiment_inventory.json و paired_comparisons/table_5_v3.csv از batch "
            f"{RUN_ID}."
        ),
        289: (
            "در اجرای نهایی، ممیزی مستقل نشان داد endpoint برچسب اجرایی در مرز کالیبراسیون ثانویه "
            "حمل و purge شده است: هر ۶۰ انتخاب outer دارای overlap_count_after_purge=0 و "
            "secondary_calibration_purge_verified=true بودند. مقایسهٔ زوجی جاری DFA1 با بدون هرست "
            "نیز با row_id و y_true یکسان، CI بلوکی ۹۵٪ و آزمون sign-flip انجام شد؛ این کنترل‌ها "
            "نتیجه را معتبرتر می‌کنند، اما چون provenance_complete=false است، آن را آزمون آینده یا "
            "مبنای ارتقای مدل نمی‌سازند."
        ),
        291: (
            "در اجرای v3 برنامهٔ زمانی معاملات غیرهم‌پوشان و benchmarkهای هزینه‌ای ثبت شد، اما همهٔ "
            "۱۲ تصمیم promotion رد شدند و منشأ داده همچنان کامل نیست. بنابراین بازده، PBO و DSR "
            "صرفاً diagnostics توسعه‌ای‌اند و هیچ ادعای سودآوری یا قابلیت معامله از این خانواده صادر نمی‌شود."
        ),
        295: (
            "برای پرسش نخست، شواهد کافی برای برتری پایدار مدل نسبت به سطح مرجع فراهم نشد. برای پرسش "
            "دوم، contrast زوجی DFA1−no-Hurst در چهار افق همگی منفی و از نظر Holm معنادار نبودند؛ "
            "فرض H0-B رد نشد. برای پرسش سوم، هیچ candidate ارتقا نیافت و شواهد برای نتیجه‌گیری اقتصادی "
            "یا استفادهٔ معاملاتی کافی نیست."
        ),
        302: (
            f"در افق یک‌روزه، دقت متوازن no-Hurst برابر {_pct(by_trial['no_hurst', 1]['pooled_balanced_accuracy'])} "
            f"و current DFA1 برابر {_pct(by_trial['current_dfa_hurst', 1]['pooled_balanced_accuracy'])} بود؛ "
            f"delta زوجی {_signed_pp(primary[1]['delta_ba'])} و CI {_decimal(primary[1]['ci_low'])} تا "
            f"{_decimal(primary[1]['ci_high'])} بود. این فاصله از نظر آماری تأیید نشد."
        ),
        303: (
            f"در افق پنج‌روزه، no-Hurst برابر {_pct(by_trial['no_hurst', 5]['pooled_balanced_accuracy'])} و "
            f"current DFA1 برابر {_pct(by_trial['current_dfa_hurst', 5]['pooled_balanced_accuracy'])} بود؛ "
            f"delta زوجی {_signed_pp(primary[5]['delta_ba'])} و Holm p={_decimal(primary[5]['p_holm'])} است. "
            f"بازوی مقاوم نقطه‌ای {_pct(by_trial['robust_hurst_regime', 5]['pooled_balanced_accuracy'])} داشت، "
            "اما CI آن نسبت به no-Hurst صفر را پوشش می‌دهد."
        ),
        304: (
            f"در افق ده‌روزه، current DFA1 با {_pct(by_trial['current_dfa_hurst', 10]['pooled_balanced_accuracy'])} "
            f"از no-Hurst با {_pct(by_trial['no_hurst', 10]['pooled_balanced_accuracy'])} پایین‌تر بود؛ "
            f"delta زوجی {_signed_pp(primary[10]['delta_ba'])} و CI {_decimal(primary[10]['ci_low'])} تا "
            f"{_decimal(primary[10]['ci_high'])} است. افت این معیار با عبور نکردن هیچ gate سازگار است."
        ),
        305: (
            f"در افق بیست‌روزه، no-Hurst برابر {_pct(by_trial['no_hurst', 20]['pooled_balanced_accuracy'])} "
            f"و current DFA1 برابر {_pct(by_trial['current_dfa_hurst', 20]['pooled_balanced_accuracy'])} بود؛ "
            f"delta زوجی {_signed_pp(primary[20]['delta_ba'])} و Holm p={_decimal(primary[20]['p_holm'])} است. "
            "بازوی مقاوم در این افق نقطه‌ای بالاتر از baseline داشت، اما CI آن صفر را پوشش می‌دهد."
        ),
        306: (
            "در مجموع، اثر DFA1 در هر چهار افق منفی است، اما هیچ‌یک از فاصله‌ها با آزمون زوجی "
            "بلوک‌آگاه و تصحیح Holm تأیید نشد. robust-Hurst نیز تنها در H1 سیگنال خامی داشت که "
            "پس از Holm به ۰٫۰۵۷۵۹۴ رسید؛ بنابراین الگوی افق‌گسترده و قابل اتکایی مشاهده نشده است."
        ),
        314: (
            "سه دسته شاهد اکنون هم‌راستا هستند. ارزیابی تاریخی قبلاً دیده شده و نقش تأییدی ندارد؛ "
            "بازاجرای توسعه‌ای v3 نقص مرز کالیبراسیون را با صفر شدن overlap در ۶۰ انتخاب اصلاح کرده؛ "
            "و paired inference، هم‌ترازی و عدم‌قطعیت تفاوت بازوها را صریحاً ثبت کرده است. بااین‌حال "
            "provenance داده کامل نیست و آزمون آینده هنوز وجود ندارد."
        ),
        315: (
            "هم‌گرایی شواهد برای رد ادعای قوی کافی است: پروژه نمی‌تواند دقت ۶۰ درصد، سودآوری قابل اتکا "
            "یا ارزش افزودهٔ پایدار هرست را اعلام کند. paired CIهای DFA1 همگی صفر را پوشش می‌دهند و "
            "هیچ trial ارتقا نیافته است. در مقابل، برای اثبات قطعی نبود هیچ سیگنال در همهٔ بازارها کافی نیست."
        ),
        316: (
            "نتیجهٔ عملی اجرای v3 توقف استقرار و نگهداری مدل‌ها در محیط پژوهش است. purge کالیبراسیون، "
            "paired inference و مستندسازی تکمیل شده‌اند؛ ادامهٔ کار باید با دادهٔ آیندهٔ واقعاً دست‌نخورده، "
            "منشأ کامل و طرحی ازپیش‌ثبت‌شده انجام شود، نه با تنظیم بیشتر روی این دوره."
        ),
        324: (
            "از دید اجرایی، تصمیم فعلی توقف استقرار و ادامهٔ کار در محیط پژوهش است. هیچ سیگنال، پروندهٔ "
            "مدل یا آستانه‌ای نباید به سامانهٔ معاملات متصل شود؛ هر ۱۲ trial رد شده‌اند و پرچم "
            "provenance_complete=false دامنهٔ تعمیم را محدود می‌کند."
        ),
        329: (
            "پژوهش حاضر فرضیهٔ ارزش افزودهٔ نمای هرست را در پیش‌بینی چندافقی جهت طلا بررسی کرد. "
            "بازاجرای نهایی v3 با ۱۲ trial نشان داد دقت متوازن current DFA1 از no-Hurst در هر چهار "
            "افق پایین‌تر است؛ deltaهای زوجی از −۰٫۹۳ تا −۳٫۲۳ واحد درصد بودند و هیچ‌کدام پس از "
            "Holm تأیید نشدند. purge مرز کالیبراسیون و paired alignment موفق بود، اما همهٔ تصمیم‌ها "
            "رد و provenance داده ناقص باقی ماند."
        ),
        333: "• اصلاح purge کالیبراسیون و بازاجرای کامل بخش توسعه انجام شد؛ تکرار بعدی باید با منشأ دادهٔ کامل باشد.",
        334: "• تقویم مشترک، مقایسهٔ زوجی و CIهای بلوکی با تصحیح Holm در v3 ثبت و اجرا شد.",
        335: "• برای آزمون آینده، bootstrap و sign-flip بلوکی با طول بلوک و بودجهٔ ازپیش‌تعریف‌شده حفظ شود.",
        340: (
            "این پروژه سامانه‌ای برای اثبات توانایی پیش‌بینی طلا ارائه نمی‌کند؛ بلکه آزمونی کنترل‌شده از "
            "یک ایدهٔ جذاب ارائه می‌دهد. پاسخ نهایی v3 منفی و محدود است: نمای DFA1 و رژیم مقاوم هرست "
            "در این داده و پروتکل ارزش افزودهٔ پیش‌بینی‌کنندهٔ پایدار نشان ندادند و هیچ مدل ارتقا نیافت. "
            "در عین حال، اصلاح purge کالیبراسیون، کنترل هم‌ترازی زوجی و گزارش دقیق عدم‌قطعیت، نتیجه را "
            "قابل ممیزی می‌کند؛ پرچم provenance ناقص مانع تعمیم آن به بازار یا آزمون آینده است."
        ),
    }
    for index, text in paragraph_updates.items():
        _replace_paragraph(document.paragraphs[index], text)

    # Table 5-2 in the revised thesis is the three-arm × four-horizon summary.
    table = document.tables[13]
    arm_rows = {
        1: "no_hurst",
        2: "current_dfa_hurst",
        3: "robust_hurst_regime",
    }
    labels = {
        "no_hurst": "بدون هرست",
        "current_dfa_hurst": "هرست مبتنی بر DFA1",
        "robust_hurst_regime": "رژیم هرست مقاوم",
    }
    for row_index, arm in arm_rows.items():
        _replace_cell(table.rows[row_index].cells[0], labels[arm])
        for col, horizon in enumerate((1, 5, 10, 20), start=1):
            _replace_cell(
                table.rows[row_index].cells[col],
                _pct(by_trial[arm, horizon]["pooled_balanced_accuracy"]),
            )
        _replace_cell(table.rows[row_index].cells[5], "هر چهار آزمایش رد؛ بدون ارتقای مدل")

    # Table 5-3 and Table 5-4 are interpretive summaries whose v2 statements are
    # no longer true after the repaired rerun.
    audit_table = document.tables[14]
    _replace_cell(
        audit_table.rows[2].cells[1], "paired DFA1 CI و Holm p ثبت شد؛ برتری پایدار مشاهده نشد"
    )
    _replace_cell(audit_table.rows[2].cells[2], "عدم مشاهدهٔ ارزش افزودهٔ پایدار")
    _replace_cell(
        audit_table.rows[3].cells[1], "purge ثانویه در ۶۰ انتخاب؛ overlap پس از purge برابر صفر"
    )
    _replace_cell(audit_table.rows[3].cells[2], "کنترل فنی موفق؛ provenance داده همچنان ناقص")
    decision_table = document.tables[15]
    _replace_cell(decision_table.rows[1].cells[1], "مجاز")
    _replace_cell(
        decision_table.rows[1].cells[2], "با paired evidence و محدودیت‌های ثبت‌شده سازگار است"
    )
    _replace_cell(decision_table.rows[3].cells[1], "غیرمجاز")
    _replace_cell(decision_table.rows[3].cells[2], "اثر DFA1 در هیچ افقی پس از Holm تأیید نشد")
    _replace_cell(decision_table.rows[4].cells[1], "غیرمجاز")
    _replace_cell(decision_table.rows[4].cells[2], "هیچ trial ارتقا نیافت؛ provenance کامل نیست")
    _replace_cell(decision_table.rows[5].cells[1], "انجام شد")
    _replace_cell(
        decision_table.rows[5].cells[2], "v3 با purge endpoint و paired inference نهایی شد"
    )

    document.save(document_path)
    _replace_ablation_chart(document_path, by_trial)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document", type=Path, default=DEFAULT_DOC)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    update_document(args.document, args.run_dir)
    print(args.document)


if __name__ == "__main__":
    main()
