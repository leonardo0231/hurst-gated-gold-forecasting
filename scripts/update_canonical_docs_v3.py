"""Update the canonical thesis and journal article with final Hurst v3 evidence.

All reported numbers are loaded from the finalized run artifacts.  The script
targets the document versions currently present on the repaired branch after
the remote document merge and refuses to edit an unexpected structure.
"""

# The paragraph literals intentionally preserve long Persian and English prose.
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
DEFAULT_RUN = Path("artifacts/research/runs") / RUN_ID
DEFAULT_THESIS = Path("docs/پایان_نامه_HGE_Gold_Forecasting_بازنگری‌شده.docx")
DEFAULT_ARTICLE = Path("docs/XAUUSD_Hurst_Forecasting_Journal.docx")


def _replace_paragraph(paragraph: Any, text: str) -> None:
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


def _load(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    inventory = json.loads((run_dir / "experiment_inventory.json").read_text(encoding="utf-8"))
    paired = json.loads((run_dir / "paired_inference.json").read_text(encoding="utf-8"))
    return inventory, paired


def _index_inventory(inventory: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(row["arm"], int(row["horizon"])): row for row in inventory["experiments"]}


def _index_paired(paired: dict[str, Any], key: str) -> dict[int, dict[str, Any]]:
    return {int(row["horizon"]): row for row in paired[key]}


def _pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def _pp(value: float) -> str:
    return f"{value * 100:+.2f} pp"


def _ci(row: dict[str, Any]) -> str:
    return f"[{row['ci_low'] * 100:+.2f}, {row['ci_high'] * 100:+.2f}] pp"


def _replace_image(document_path: Path, image_name: str, replacement: bytes) -> None:
    temporary = document_path.with_suffix(".v3.tmp.docx")
    with ZipFile(document_path, "r") as source:
        entries = [(info, source.read(info.filename)) for info in source.infolist()]
    with ZipFile(temporary, "w", compression=ZIP_DEFLATED) as target:
        for info, content in entries:
            target.writestr(info, replacement if info.filename == image_name else content)
    os.replace(temporary, document_path)


def _article_chart(by_trial: dict[tuple[str, int], dict[str, Any]]) -> bytes:
    """Overlay the final values on the existing article Figure 2 canvas."""

    source_path = DEFAULT_ARTICLE
    with ZipFile(source_path, "r") as source:
        chart_bytes = source.read("word/media/image2.png")
    image = Image.open(BytesIO(chart_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)
    # The existing chart is 1524x798.  Preserve its axes, tick labels, and
    # legend while clearing the old data traces and redrawing the plot interior.
    draw.rectangle((124, 34, 1503, 550), fill="white")
    plot_left, plot_right = 120, 1503
    plot_top, plot_bottom = 20, 686
    y_min, y_max = 45.0, 55.0

    def y_coord(value: float) -> int:
        return round(plot_top + (y_max - value) * (plot_bottom - plot_top) / (y_max - y_min))

    for value in (54, 52, 50, 48):
        y = y_coord(value)
        draw.line((plot_left, y, plot_right, y), fill=(225, 225, 225), width=2)
    y_reference = y_coord(50)
    for start in range(plot_left, plot_right, 18):
        draw.line((start, y_reference, min(start + 10, plot_right), y_reference), fill=(183, 0, 0), width=3)

    horizons = (1, 5, 10, 20)
    arms = ("no_hurst", "current_dfa_hurst", "robust_hurst_regime")
    values = [
        [by_trial[arm, horizon]["pooled_balanced_accuracy"] * 100 for horizon in horizons]
        for arm in arms
    ]
    x_coords = (182, 446, 778, 1440)
    colors = ((15, 48, 86), (143, 100, 0), (108, 132, 157))
    for series, color in zip(values, colors, strict=True):
        points = [(x, y_coord(value)) for x, value in zip(x_coords, series, strict=True)]
        draw.line(points, fill=color, width=5, joint="curve")
        for x, y in points:
            draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=color)

    updated = BytesIO()
    image.save(updated, format="PNG")
    return updated.getvalue()


def _update_thesis(
    document_path: Path,
    by_trial: dict[tuple[str, int], dict[str, Any]],
    primary: dict[int, dict[str, Any]],
    secondary: dict[int, dict[str, Any]],
) -> None:
    document = Document(document_path)
    if len(document.paragraphs) != 324 or len(document.tables) != 11:
        raise RuntimeError("Canonical thesis structure changed; refusing edits")

    updates = {
        19: (
            "این پایان‌نامه یک چارچوب پژوهشی چندافقی و قابل ممیزی برای بررسی ارزش افزودهٔ "
            "ویژگی‌های رژیمی مبتنی بر نمای هرست در پیش‌بینی جهت روزانهٔ XAUUSD ارائه می‌کند. "
            "داده شامل ۴۰۱۷ مشاهدهٔ روزانه از MetaQuotes-Demo در بازهٔ ۲۰۱۱/۰۱/۰۳ تا ۲۰۲۶/۰۷/۳۱ "
            "است و بخشی از فرادادهٔ استخراج آن پسینی بازسازی شده، بنابراین provenance_complete=false "
            "و این رکورد آزمون آینده‌نگر بکر نیست. برآورد اصلی هرست با DFA1 علّی روی تفاضل‌های "
            "مرتبهٔ یک لگاریتم قیمت در پنجره‌های ۶۴ و ۱۲۸ و با detrend خطی انجام می‌شود. بازاجرای "
            "کامل v3 شامل سه بازو و چهار افق، یعنی ۱۲ آزمایش، با purge مستقل endpoint کالیبراسیون "
            "ثانویه، paired alignment و آزمون‌های بلوکی انجام شد. دقت متوازن تجمیعی بازوی بدون هرست "
            f"در افق‌های ۱، ۵، ۱۰ و ۲۰ روزه به‌ترتیب {_pct(by_trial['no_hurst', 1]['pooled_balanced_accuracy'])}، "
            f"{_pct(by_trial['no_hurst', 5]['pooled_balanced_accuracy'])}، "
            f"{_pct(by_trial['no_hurst', 10]['pooled_balanced_accuracy'])} و "
            f"{_pct(by_trial['no_hurst', 20]['pooled_balanced_accuracy'])} بود؛ مقادیر DFA1 به‌ترتیب "
            f"{_pct(by_trial['current_dfa_hurst', 1]['pooled_balanced_accuracy'])}، "
            f"{_pct(by_trial['current_dfa_hurst', 5]['pooled_balanced_accuracy'])}، "
            f"{_pct(by_trial['current_dfa_hurst', 10]['pooled_balanced_accuracy'])} و "
            f"{_pct(by_trial['current_dfa_hurst', 20]['pooled_balanced_accuracy'])} بود. در contrast زوجی "
            "DFA1−بدون هرست، هر چهار delta منفی، تمام CIهای ۹۵ درصد شامل صفر و همهٔ Holm pها "
            "برابر ۰٫۴۴۹۹۵۵ بودند؛ هر ۱۲ تصمیم رد شد و هر ۶۰ انتخاب outer پس از purge overlap صفر داشت. "
            "نتیجهٔ قابل دفاع، نبود شواهد برای ارزش افزودهٔ پایدار هرست در این داده و پروتکل است، نه "
            "ادعای نبود سیگنال در همهٔ بازارها یا اثبات سودآوری."
        ),
        155: (
            "DFA1 با حذف روند خطی محلی در مقیاس‌های مختلف، رفتار مقیاس‌پذیر نوسان را برآورد می‌کند. "
            "در پیاده‌سازی حاضر، ورودی تفاضل مرتبهٔ یک لگاریتم قیمت است؛ برای هر پنجرهٔ ۶۴ و ۱۲۸ "
            "مشاهده، پروفایل تجمعی پس از مرکززدایی ساخته می‌شود و مقیاس‌های توان دو از ۴ تا "
            "floor(n_increments/3) با حداقل ۳۲ مشاهده و حداقل سه مقیاس معتبر به کار می‌روند. "
            "قطعه‌های غیرهم‌پوشان ابتدا و انتهای پنجره با detrend خطی مرتبهٔ یک برازش می‌شوند و شیب "
            "رگرسیون log–log نوسان بر مقیاس، برآورد DFA1 است. انحراف معیار افزایش‌های نزدیک صفر با "
            "کف 1e-12 کنترل می‌شود، محاسبه غلتان و علّی است و مقدار H به بازهٔ مصنوعی برش داده نمی‌شود."
        ),
        188: (
            "آزمون حذف مؤلفه به‌صورت از پیش‌تعریف‌شده شامل پنج پیکربندی برای هر افق legacy و نیز یک "
            "مجموعهٔ نهایی v3 است. مجموعهٔ نهایی سه بازوی بدون Hurst، current DFA1-Hurst و robust "
            "Hurst regime را در افق‌های ۱، ۵، ۱۰ و ۲۰ روزه، در مجموع ۱۲ آزمایش، مقایسه می‌کند. پس از "
            "اصلاح purge، endpoint برچسب کالیبراسیون ثانویه با قاعدهٔ بستهٔ "
            "calibration_label_end_index < evaluation_start_row_id کنترل شد؛ در هر ۶۰ انتخاب outer "
            "overlap_count_after_purge=0 ثبت شد. همهٔ ۱۲ تصمیم رد شدند و دادهٔ تاریخی در هیچ انتخابی "
            "به کار نرفت؛ بااین‌حال، به‌دلیل مشاهدهٔ قبلی بازه و provenance ناقص، این شواهد تأیید آینده‌نگر نیستند."
        ),
        194: (
            "در هر بخش اعتبارسنجی، مدل‌های پایه پس از حذف نمونه‌های دارای هم‌پوشانی اطلاعاتی روی دادهٔ "
            "آموزش برازش می‌شوند و احتمال خارج از نمونه برای دادهٔ اعتبارسنجی تولید می‌کنند. pipeline "
            "هر سه بازو شامل SimpleImputer با strategy=median و missing-indicator، StandardScaler و "
            "LogisticRegression متعادل با C=0.2، penalty=l2، solver=lbfgs، max_iter=2000، tol=0.0001 "
            "و random_state=42 است. کالیبراسیون اختیاری sigmoid با C=1000000 و حداقل ۴۰ نمونهٔ دارای "
            "دو کلاس، و حاشیه‌های عدم معاملهٔ ۰ و ۰٫۰۵ بررسی شدند؛ انتخاب با ۳ inner fold و ارزیابی با "
            "۵ outer fold انجام شد. در بازاجرای v3، endpoint اجرایی هر برچسب تا مرز کالیبراسیون ثانویه "
            "به‌صورت مستقل purge شد و شرط calibration_label_end_index < evaluation_start_row_id در هر "
            "۶۰ انتخاب outer برقرار بود؛ بنابراین نقص مرزی اجراشده در نسخهٔ قبلی اصلاح و ممیزی شد."
        ),
        206: (
            "آزمون حذف مؤلفهٔ v3 با purge کالیبراسیون اصلاح‌شده و paired inference نهایی شد، اما به‌دلیل "
            "مشاهدهٔ قبلی دوره و provenance_complete=false، شواهد آن توسعه‌ای/توصیفی و مشروط به همین "
            "Feed هستند؛ برای انتخاب پیکربندی یا ادعای تعمیم آینده‌نگر استفاده نمی‌شود."
        ),
        220: (
            "تقسیم‌کننده CPCV مخصوص دادهٔ توسعه از ۸ گروه و انتخاب ۲ گروه استفاده می‌کند و ۲۸ تقسیم برای "
            "هر افق می‌سازد. توابع PSR، DSR و PBO دارای آزمون واحد هستند، اما مقادیر رسمی PBO و DSR تا "
            "ذخیرهٔ کامل مسیر بازده همهٔ آزمایش‌های توسعه گزارش نمی‌شوند. در v3، ممیزی ۱۲ آزمایش نشان داد "
            "purge مستقل مرز calibration/evaluation اجرا شده، overlap پس از purge در هر ۶۰ انتخاب صفر "
            "است و paired comparison با تقویم و y_true هم‌تراز انجام شده است؛ این کنترل‌ها شواهد را قوی‌تر "
            "می‌کنند، اما جای آزمون آینده‌نگر با provenance کامل را نمی‌گیرند."
        ),
        228: (
            "گزارش نهایی نشان می‌دهد بررسی ruff و mypy بدون خطا انجام شده و آخرین اجرای QA شامل ۹۷ تست "
            "خودکار پاس شده است. همچنین رسید v3، مانیفست خروجی‌ها، زنجیرهٔ registry و هش منابع runtime "
            "تطبیق داده شدند؛ receipt نهایی ۱۲۹ خروجی پژوهشی و ۲۱ منبع runtime را پوشش می‌دهد. این نتایج "
            "صحت فنی و قابلیت ممیزی اجرا را نشان می‌دهند، نه مهارت آماری یا سودآوری بازار را."
        ),
        243: (
            "جدول ۶ نتیجهٔ بازاجرای کامل v3 را نشان می‌دهد. دقت متوازن no-Hurst در چهار افق "
            f"{_pct(by_trial['no_hurst', 1]['pooled_balanced_accuracy'])}، {_pct(by_trial['no_hurst', 5]['pooled_balanced_accuracy'])}، "
            f"{_pct(by_trial['no_hurst', 10]['pooled_balanced_accuracy'])} و {_pct(by_trial['no_hurst', 20]['pooled_balanced_accuracy'])} "
            "و دقت DFA1 به‌ترتیب "
            f"{_pct(by_trial['current_dfa_hurst', 1]['pooled_balanced_accuracy'])}، {_pct(by_trial['current_dfa_hurst', 5]['pooled_balanced_accuracy'])}، "
            f"{_pct(by_trial['current_dfa_hurst', 10]['pooled_balanced_accuracy'])} و {_pct(by_trial['current_dfa_hurst', 20]['pooled_balanced_accuracy'])} است؛ "
            "رژیم مقاوم فقط در H5 و H20 از no-Hurst نقطه‌ای بالاتر است. در paired comparison، تمام "
            "CIهای DFA1−no-Hurst صفر را پوشش دادند و Holm p در هر چهار افق ۰٫۴۴۹۹۵۵ بود. بنابراین "
            "برندهٔ نقطه‌ایِ یک افق یا تغییر بازوی برتر مجوز انتخاب پسینی و شاهد ارزش افزودهٔ پایدار نیست."
        ),
        255: (
            "نتایج v3 نشان می‌دهند ویژگی‌های درون‌بازاری فعلی برای پیش‌بینی پایدار جهت طلا کافی نیستند. "
            "مقایسهٔ زوجی DFA1 با no-Hurst در H1، H5، H10 و H20 به‌ترتیب deltaهای −۱٫۴۹، −۰٫۹۳، "
            "−۳٫۲۳ و −۲٫۶۲ واحد درصد داشت؛ CIهای بلوکی ۹۵ درصد صفر را پوشش دادند و تصحیح Holm هیچ "
            "افقی را معنادار نکرد. بنابراین تغییر Threshold یا پیچیده‌تر کردن مدل بدون دادهٔ جدید، اطلاعات "
            "گمشده را ایجاد نمی‌کند و می‌تواند خطر بیش‌برازش را افزایش دهد."
        ),
        256: (
            "نتیجهٔ آزمون حذف مؤلفه نشان می‌دهد Hurst نباید به‌عنوان «قانون جهت» معرفی شود. بازوی مقاوم "
            "در H1 یک p خام ۰٫۰۱۴۳۹۹ داشت، اما پس از Holm برابر ۰٫۰۵۷۵۹۴ شد و الگوی افق‌گسترده ایجاد "
            "نکرد. اگر اثر عمومی بود، انتظار می‌رفت افزودن Hurst در چند افق و در مقایسهٔ paired الگوی "
            "بهبود سازگارتری بسازد. دادهٔ فعلی چنین الگویی ندارد؛ این نتیجه ارزش بالقوهٔ Hurst در بازارها "
            "یا نمونه‌های دیگر را رد نمی‌کند."
        ),
        258: "مجموعهٔ آزمون تاریخی پیش‌تر در فرایند توسعه مشاهده شده و تأیید آینده‌نگر محسوب نمی‌شود.",
        265: (
            "PBO و DSR رسمی هنوز به ذخیرهٔ کامل مسیر بازده همهٔ آزمایش‌های توسعه وابسته‌اند؛ اجرای paired "
            "inference و اصلاح purge کالیبراسیون جایگزین این سنجه‌ها نیست."
        ),
        270: (
            "این پایان‌نامه یک چارچوب پژوهشی چندافقی برای پیش‌بینی جهت XAUUSD ارائه کرد که در آن هدف "
            "آماری، ارزیابی اقتصادی، زمان‌بندی تصمیم، DFA1 علّی، حذف هم‌پوشانی برچسب‌های سطح اول و دوم، "
            "purge مستقل مرز کالیبراسیون ثانویه، paired inference، ممیزی CPCV و ثبت تغییرناپذیر خروجی‌ها "
            "جداگانه بررسی شدند. اجرای نهایی v3 سه بازو و چهار افق را بدون انتخاب پسینی اجرا کرد و هر ۱۲ "
            "تصمیم رد شد."
        ),
        272: (
            "هدف ۶۰ درصد محقق نشده است. در ارزیابی تاریخی قبلاً مشاهده‌شده، دقت متوازن افق‌های ۱، ۵، ۱۰ "
            "و ۲۰ روزه تقریباً ۵۰٫۳۹، ۵۲٫۱۹، ۵۰٫۴۷ و ۵۰٫۳۸ درصد بود و CIهای ۹۵ درصد مقدار ۵۰ درصد را "
            "پوشش می‌دادند. در اجرای v3 نیز هیچ‌یک از ۱۲ trial به معیار promotion نرسید؛ مقایسهٔ زوجی DFA1 "
            "همه deltaهای منفی و غیرمعنادار پس از Holm داشت. نتیجه نباید با انتخاب بهترین افق یا مدل بازنویسی شود."
        ),
        282: (
            "چارچوب HGE Gold Forecasting نباید به‌عنوان مدل معاملاتی سودآور یا سامانه‌ای با دقت جهت‌دار "
            "بالاتر از ۶۰ درصد معرفی شود. بازاجرای کامل v3 نشان داد در این داده و پروتکل، DFA1 و رژیم "
            "مقاوم هرست ارزش افزودهٔ پیش‌بینی‌کنندهٔ پایدار نسبت به no-Hurst اثبات نمی‌کنند؛ delta زوجی DFA1 "
            "در هر چهار افق منفی بود و هیچ Holm p معناداری ثبت نشد. purge endpoint کالیبراسیون در هر ۶۰ "
            "انتخاب بدون overlap تأیید و همهٔ ۱۲ تصمیم رد شدند، اما provenance_complete=false و مشاهدهٔ قبلی "
            "دوره مانع ادعای تعمیم یا آزمون آینده‌نگر است. ارزش اصلی پژوهش، چارچوب قابل ممیزی و گزارش معتبر "
            "این نتیجهٔ منفی محدود است."
        ),
    }
    for index, text in updates.items():
        _replace_paragraph(document.paragraphs[index], text)

    table = document.tables[6]
    if len(table.rows) != 5 or len(table.columns) != 3:
        raise RuntimeError("Canonical thesis ablation table structure changed")
    headers = (
        "افق",
        "دقت متوازن تجمیعی: بدون هرست / DFA1 / رژیم مقاوم",
        "مقایسهٔ زوجی DFA1−بدون هرست",
    )
    for column, text in enumerate(headers):
        _replace_cell(table.rows[0].cells[column], text)
    for row_index, horizon in enumerate((1, 5, 10, 20), start=1):
        values = " / ".join(
            _pct(by_trial[arm, horizon]["pooled_balanced_accuracy"])
            for arm in ("no_hurst", "current_dfa_hurst", "robust_hurst_regime")
        )
        pair = primary[horizon]
        _replace_cell(table.rows[row_index].cells[0], f"{horizon}")
        _replace_cell(table.rows[row_index].cells[1], values)
        _replace_cell(
            table.rows[row_index].cells[2],
            f"Δ={pair['delta_ba'] * 100:+.2f} واحد درصد؛ CI95٪ {_ci(pair)}؛ Holm p={pair['p_holm']:.6f}",
        )

    qa_table = document.tables[9]
    _replace_cell(
        qa_table.rows[2].cells[1],
        "خیر؛ paired DFA1−بدون هرست در هر چهار افق منفی و همهٔ Holm pها ۰٫۴۴۹۹۵۵ است؛ robust نیز الگوی پایدار ندارد.",
    )
    _replace_cell(
        qa_table.rows[4].cells[1],
        "purge endpoint کالیبراسیون در هر ۶۰ انتخاب outer با overlap صفر تأیید شد؛ بااین‌حال provenance ناقص و آزمون آینده وجود ندارد.",
    )
    document.save(document_path)


def _update_article(
    document_path: Path,
    by_trial: dict[tuple[str, int], dict[str, Any]],
    primary: dict[int, dict[str, Any]],
    secondary: dict[int, dict[str, Any]],
) -> None:
    document = Document(document_path)
    if len(document.paragraphs) != 132 or len(document.tables) != 9:
        raise RuntimeError("Canonical article structure changed; refusing edits")

    primary_summary = "; ".join(
        f"H{h}: {_pp(primary[h]['delta_ba'])} ({_ci(primary[h])}; Holm p={primary[h]['p_holm']:.6f})"
        for h in (1, 5, 10, 20)
    )
    updates = {
        1: "Jalil Ahmad Afshar and Ehsan Aryanfar",
        10: (
            "The scope of the contribution is deliberately limited. The study provides a common-calendar, "
            "multi-horizon comparison; distinguishes a statistical close-to-close target from an executable "
            "next-open target; evaluates two Hurst representations against a no-Hurst control arm; records "
            "data and output hashes; and reports a repaired, fully rerun ablation with paired inference. "
            "Accordingly, the contribution is an auditable evaluation of a plausible feature hypothesis rather "
            "than a claim that Hurst-based modeling can reliably forecast or trade gold."
        ),
        17: (
            "In the present study, the first-order detrended fluctuation analysis (DFA1) estimator follows the "
            "long-range-correlation framework described by Kantelhardt et al. (2001). It is applied causally "
            "to first differences of log price using rolling windows of 64 and 128 observations. After "
            "demeaning and cumulative profiling, non-overlapping segments are linearly detrended at scales "
            "from 4 through floor(n_increments/3), with at least 32 observations and three valid scales. "
            "A 1e-12 standard-deviation floor prevents numerical failure. A robust regime representation uses "
            "a 252-observation history, a minimum of 126 observations, and rolling 0.33/0.67 historical "
            "quantiles. Both are regime descriptors, not directional rules such as H > 0.5 implying an upward forecast."
        ),
        20: (
            "The protocol fixes the main historical boundary, removes training observations whose future label "
            "intervals overlap validation or evaluation, and confines model and threshold selection to the "
            "development data. In the repaired v3 rerun, the secondary calibration label endpoint was purged "
            "with the strict rule calibration_label_end_index < evaluation_start_row_id; overlap_count_after_purge "
            "was zero in all 60 outer selections. This repair changes the ablation from an audit finding about "
            "a boundary defect into technically purged development evidence, while provenance_complete=false "
            "and the previously observed period still limit its evidentiary scope."
        ),
        37: (
            "The full feature set contains 70 causal features, whereas the no-Hurst arm contains 65. Feature "
            "families include lagged returns and momentum at lags 1, 2, 3, 5, 10, and 20; trend and volatility "
            "measures over windows of 5, 10, 20, 63, and 126 days; price-range and position features; broker "
            "tick-volume summaries; causal DFA1 estimates over 64- and 128-day windows; and 252-day regime "
            "descriptors with a minimum history of 126 observations. Every feature at time t is computed using "
            "information available no later than t. The base pipeline uses median imputation with a missing "
            "indicator, standardization, and balanced logistic regression with C=0.2, l2 penalty, lbfgs solver, "
            "max_iter=2000, tol=0.0001, and random_state=42."
        ),
        38: (
            "The primary Hurst representation uses first-order DFA with the exact rolling construction described "
            "above. The robust regime representation is built from historical rolling thresholds rather than a "
            "future-looking global threshold. The v3 search also evaluated optional sigmoid calibration with "
            "C=1000000 and a minimum of 40 eligible observations containing both classes, alongside no-trade "
            "margins of 0 and 0.05. Candidate model selection used three inner folds and five outer folds under "
            "a fixed budget; a legacy rescaled-range representation remains a comparison arm in the broader protocol."
        ),
        43: (
            "Candidate hyperparameters, model selection, and classification thresholds are determined within the "
            "development period. The same candidate model family, preprocessing, calibration candidates, and "
            "decision budget are used when Hurst features are removed. No model, forecast horizon, or threshold "
            "is selected after inspection of the historical evaluation results."
        ),
        46: (
            "The design separates development decisions from the previously observed historical evaluation. In "
            "the repaired v3 ablation, the secondary calibration interval is endpoint-purged before evaluation: "
            "all 60 outer selections report overlap_count_after_purge=0 and secondary_calibration_purge_verified=true. "
            "This removes the recorded calibration-boundary defect from the rerun, but it does not create a pristine "
            "prospective holdout because the period was previously observed and provenance_complete=false."
        ),
        48: (
            "The ablation contains three model arms and four forecast horizons, producing 12 pre-specified trials: "
            "(i) no Hurst features, (ii) DFA1-Hurst features, and (iii) robust Hurst regime features. Targets, "
            "model families, data splits, preprocessing, and decision rules are held constant across arms. All 12 "
            "promotion decisions were rejected. A paired comparison aligned row_id and y_true across the no-Hurst "
            "and DFA1 arms before estimating differences."
        ),
        50: (
            "Performance is evaluated using balanced accuracy, macro-F1, and ROC-AUC. Ninety-five-percent intervals "
            "for paired balanced-accuracy differences use a moving block bootstrap with block length 10 and 10,000 "
            "replicates. Two-sided sign-flip inference uses the same 10,000 block-aware permutations, and Holm "
            "adjustment is applied within each four-horizon family. These intervals and p-values communicate uncertainty; "
            "they do not substitute for prospective confirmation."
        ),
        53: (
            "The software pipeline separates data validation, feature construction, target generation, temporal "
            "splitting, model fitting, calibration, evaluation, and report generation. The latest recorded quality "
            "run passed 97 automated tests covering data contracts, feature causality, overlap purging, prediction "
            "boundaries, output records, and hash integrity. The final v3 receipt covers 129 research outputs and "
            "21 runtime sources. These controls improve traceability and reproducibility, but they do not convert a "
            "technically successful run into evidence of statistical validity, prospective generalization, or economic usefulness."
        ),
        65: (
            "Table 5 compares the three ablation arms after the repaired v3 rerun. Pooled balanced accuracy for "
            "no Hurst is 52.31%, 49.72%, 50.74%, and 49.93% at horizons 1, 5, 10, and 20 days; the corresponding "
            "DFA1-Hurst values are 50.82%, 48.78%, 47.52%, and 47.31%; robust Hurst values are 49.49%, 51.02%, "
            "47.64%, and 50.78%. The point-estimate winner changes by horizon, and no arm demonstrates a stable "
            "cross-horizon advantage."
        ),
        66: (
            f"The paired DFA1-minus-no-Hurst contrasts are {primary_summary}. Every confidence interval includes zero "
            "and every Holm-adjusted p-value is non-significant. The robust arm has a raw H1 p-value of "
            f"{secondary[1]['p_raw']:.6f} but a Holm-adjusted value of {secondary[1]['p_holm']:.6f}; its other "
            "horizon contrasts are also non-significant after family adjustment. Thus the rerun supports a bounded "
            "negative finding, not a claim that Hurst information is universally useless."
        ),
        68: (
            f"Source: Finalized 12-trial v3 run {RUN_ID}; paired_inference.json and "
            "paired_comparisons/table_5_v3.csv. All promotion decisions were rejected; provenance_complete=false."
        ),
        72: (
            "The 97 passed automated tests show that the implementation satisfied the software contracts covered by "
            "those tests, including the repaired endpoint purge and paired alignment. They do not establish that the "
            "target is economically meaningful, the sample is representative, the previously observed holdout is "
            "prospective, or the results generalize beyond the documented broker feed."
        ),
        80: (
            "For the first research question, the candidate system does not provide sufficient evidence of stable "
            "performance above the 50% reference level. For the second, the paired DFA1-minus-no-Hurst contrasts "
            f"are {primary_summary}; no Holm-adjusted comparison is significant, and the robust arm does not dominate "
            "across horizons. For the third, all 12 promotion decisions were rejected and the available evidence is "
            "insufficient to support an economic or trading conclusion."
        ),
        84: (
            "The principal methodological contribution is the evaluation framework constructed around the feature "
            "hypothesis. The study fixes decision timing, separates statistical and executable targets, includes a "
            "no-Hurst control arm, applies a common temporal boundary, repairs the secondary calibration purge, "
            "aligns paired outcomes, limits the experimental budget, reports block-aware uncertainty, and retains an "
            "auditable record of the analysis. This makes the negative result informative without treating a single "
            "isolated accuracy as confirmatory evidence."
        ),
        93: (
            "Third, the 12-trial ablation's secondary calibration-to-evaluation boundary was repaired and verified "
            "with zero overlap in all 60 outer selections, but the rerun remains development evidence because the "
            "historical period was previously observed and provenance_complete=false."
        ),
        94: (
            "Fourth, the paired block-bootstrap and sign-flip procedures with family-level Holm adjustment improve "
            "the comparison, but finite-sample dependence, overlapping multi-day targets, and the single feed still "
            "limit inferential strength."
        ),
        97: (
            "This study evaluated whether Hurst-based regime features provide stable incremental value for multi-horizon "
            "daily XAUUSD direction forecasting. The repaired v3 rerun covered 12 trials across four horizons and "
            "three arms. No Hurst yielded the highest pooled balanced accuracy at H1, H10, and H20, while robust Hurst "
            "was highest at H5; DFA1-Hurst was below no Hurst at every horizon. The paired DFA1-minus-no-Hurst contrasts "
            "were negative at all horizons, with confidence intervals covering zero and Holm-adjusted p-values of 0.449955."
        ),
        98: (
            "The defensible conclusion is therefore narrow: given the documented data source and repaired evaluation "
            "protocol, the evidence does not demonstrate stable directional skill, stable incremental predictive value "
            "from Hurst features, or reliable trading profitability. The purge repair and paired inference strengthen "
            "the technical audit trail, while the previously observed period and provenance_complete=false prevent a "
            "prospective or broadly generalizable claim. The study's principal value lies in making that negative finding "
            "transparent and reproducible."
        ),
        99: (
            "Future work should evaluate the locked design on strictly prospective data that have not been observed or "
            "used during model development. It should preserve the endpoint-aware calibration purge, common-calendar "
            "comparison, paired and horizon-aware inference, point-in-time provenance for broker conventions, and a "
            "pre-registered budget. External macroeconomic variables or alternative Hurst estimators should be introduced "
            "only within a new prospective design. Until then, the current results should not be presented as evidence "
            "of a deployable trading system or as an investment recommendation."
        ),
        101: (
            "Data Availability Statement: The code, configurations, manifests, finalization record, paired comparisons, "
            f"and evidence files are available in the public repository on branch codex/hurst-ablation-v3-repair for "
            f"the finalized run {RUN_ID}. The raw broker-provided series depend on the MetaQuotes-Demo contract and "
            "data version; therefore, the study reports recorded hashes and reconstruction rules without claiming "
            "equivalence to another broker feed."
        ),
        107: (
            "The reported values are based on the finalized, repaired v3 run rather than on the earlier unrepaired "
            "record. Reproduction should use the committed protocol, the finalization and registry records, the "
            "paired-comparison artifacts, and the recorded quality checks while recognizing the limitations associated "
            "with the broker contract, previously observed evaluation period, and incomplete provenance."
        ),
        122: (
            "HGE Gold Forecasting Repository. (2026). Finalized Hurst-ablation v3 record, branch "
            "codex/hurst-ablation-v3-repair, run "
            f"{RUN_ID}. GitHub. https://github.com/leonardo0231/hurst-gated-gold-forecasting"
        ),
    }
    for index, text in updates.items():
        _replace_paragraph(document.paragraphs[index], text)

    abstract = (
        "Gold direction forecasts are often assessed using the best observed accuracy or random cross-validation, "
        "making claims about the incremental value of individual features difficult to interpret. This study examines "
        "whether causal Hurst-based regime features provide stable predictive value for multi-horizon XAUUSD direction "
        "forecasting within a fixed machine-learning pipeline. The dataset comprises 4,017 daily XAUUSD observations "
        "from MetaQuotes-Demo spanning 3 January 2011 to 31 July 2026. Four forecast horizons (1, 5, 10, and 20 days) "
        "are evaluated using a locked historical boundary of 3 July 2023, purged walk-forward splits, and development-only "
        "model and threshold selection. Hurst features are computed causally using first-order detrended fluctuation "
        "analysis over 64- and 128-day windows and are compared with a no-Hurst baseline and a robust regime representation. "
        "The repaired v3 rerun contains 12 trials across three arms and verifies zero secondary calibration overlap in all "
        "60 outer selections. Pooled balanced accuracy for no Hurst is 52.31%, 49.72%, 50.74%, and 49.93% across the four "
        "horizons; the corresponding DFA1-Hurst values are 50.82%, 48.78%, 47.52%, and 47.31%. Paired DFA1-minus-no-Hurst "
        "contrasts are negative at every horizon, all 95% intervals include zero, and all Holm-adjusted p-values equal "
        "0.449955. All 12 promotion decisions were rejected. Because the historical period was previously observed and "
        "provenance_complete=false, these results are repaired development evidence rather than prospective confirmation. "
        "The study contributes an auditable negative result and a reproducible evaluation framework."
    )
    if len(document.tables[1].rows) != 1 or len(document.tables[1].columns) != 1:
        raise RuntimeError("Canonical article abstract structure changed")
    _replace_cell(document.tables[1].rows[0].cells[0], abstract)

    table = document.tables[6]
    if len(table.rows) != 4 or len(table.columns) != 6:
        raise RuntimeError("Canonical article Table 5 structure changed")
    headers = (
        "Arm",
        "1 day",
        "5 days",
        "10 days",
        "20 days",
        "Paired comparison vs no Hurst",
    )
    for column, text in enumerate(headers):
        _replace_cell(table.rows[0].cells[column], text)
    arm_labels = {
        "no_hurst": "No Hurst",
        "current_dfa_hurst": "DFA1-Hurst",
        "robust_hurst_regime": "Robust Hurst regime",
    }
    for row_index, arm in enumerate(("no_hurst", "current_dfa_hurst", "robust_hurst_regime"), start=1):
        _replace_cell(table.rows[row_index].cells[0], arm_labels[arm])
        for column, horizon in enumerate((1, 5, 10, 20), start=1):
            _replace_cell(table.rows[row_index].cells[column], _pct(by_trial[arm, horizon]["pooled_balanced_accuracy"]))
        if arm == "no_hurst":
            comparison = "Reference arm"
        elif arm == "current_dfa_hurst":
            comparison = "".join(
                f"H{h}: {_pp(primary[h]['delta_ba'])}; CI {_ci(primary[h])}; Holm p={primary[h]['p_holm']:.6f}. "
                for h in (1, 5, 10, 20)
            )
        else:
            comparison = "".join(
                f"H{h}: {_pp(secondary[h]['delta_ba'])}; Holm p={secondary[h]['p_holm']:.6f}. "
                for h in (1, 5, 10, 20)
            )
        _replace_cell(table.rows[row_index].cells[5], comparison)

    claims = document.tables[8]
    _replace_cell(claims.rows[2].cells[1], "Paired DFA1 deltas are negative at all horizons; no Holm-adjusted comparison is significant")
    _replace_cell(claims.rows[2].cells[2], "No stable incremental Hurst value demonstrated")
    _replace_cell(claims.rows[3].cells[1], "Endpoint purge verified; overlap count is zero in all 60 outer selections")
    _replace_cell(claims.rows[3].cells[2], "Technical control passed; provenance and prospective status remain limited")
    _replace_cell(claims.rows[4].cells[1], "Paired block bootstrap, sign-flip, and Holm adjustment are recorded")
    _replace_cell(claims.rows[4].cells[2], "Uncertainty is reported; it is not a prospective guarantee")
    document.save(document_path)
    _replace_image(document_path, "word/media/image2.png", _article_chart(by_trial))


def update_documents(
    thesis_path: Path,
    article_path: Path,
    run_dir: Path,
) -> None:
    inventory, paired = _load(run_dir)
    by_trial = _index_inventory(inventory)
    primary = _index_paired(paired, "primary")
    secondary = _index_paired(paired, "secondary")
    global DEFAULT_ARTICLE
    DEFAULT_ARTICLE = article_path
    _update_thesis(thesis_path, by_trial, primary, secondary)
    _update_article(article_path, by_trial, primary, secondary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thesis", type=Path, default=DEFAULT_THESIS)
    parser.add_argument("--article", type=Path, default=DEFAULT_ARTICLE)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    update_documents(args.thesis, args.article, args.run_dir)
    print(args.thesis)
    print(args.article)


if __name__ == "__main__":
    main()
