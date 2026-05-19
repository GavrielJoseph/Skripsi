"""
scraping.py — Tokopedia Review Scraper
Dipanggil oleh Laravel AnalysisController via shell_exec:
  python scraping.py <url> <product_name> <max_pages> <output_csv_path>
"""

import sys
import time
import json
import pandas as pd
import re
import io
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# Paksa stdout untuk menggunakan UTF-8 agar tidak crash di Windows saat print emoji
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def setup_driver():
    options = Options()
    
    # Mode headless dimatikan agar bisa lihat progress di layar
    # options.add_argument("--headless=new") 
    
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver


def slow_scroll_to(driver, pct):
    driver.execute_script(f"window.scrollTo(0, document.body.scrollHeight * {pct});")
    time.sleep(0.5)


def scrape_reviews(url, product_name, max_pages=50):
    driver = setup_driver()
    all_reviews = []

    try:
        print(f"Memproses: {product_name}", flush=True)
        driver.get(url)
        time.sleep(8)

        # Scroll perlahan agar elemen me-load
        for pct in [0.2, 0.4, 0.6, 0.8, 1.0]:
            slow_scroll_to(driver, pct)
        time.sleep(3)

        # Cari tab ulasan jika belum di halaman /review
        if "/review" not in url:
            print("Mencari tab Ulasan...", flush=True)
            try:
                tab = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH,
                        "//*[contains(text(),'Ulasan') and "
                        "(self::button or self::a or self::li or self::div)]"))
                )
                driver.execute_script("arguments[0].scrollIntoView(true);", tab)
                time.sleep(1)
                driver.execute_script("arguments[0].click();", tab)
                time.sleep(5)
                print("Tab Ulasan diklik", flush=True)
            except:
                print("Gagal klik tab, mencoba lanjut scrape langsung...", flush=True)

        # Loop filter bintang 5 -> 1
        for star in [5, 4, 3, 2, 1]:
            print(f"Mengaktifkan filter Bintang {star}...", flush=True)

            # Klik checkbox bintang
            try:
                filter_xpath = f"//span[text()='{star}']/ancestor::label"
                filter_el = WebDriverWait(driver, 5).until(
                    EC.presence_of_element_located((By.XPATH, filter_xpath))
                )
                driver.execute_script(
                    "arguments[0].scrollIntoView({block: 'center'});", filter_el
                )
                time.sleep(1)
                driver.execute_script("arguments[0].click();", filter_el)
                time.sleep(4) 
                print(f"Filter Bintang {star} berhasil diklik", flush=True)
            except Exception as e:
                print(f"Gagal klik filter bintang {star}. Melewati...", flush=True)
                continue

            # Loop halaman untuk bintang terpilih
            for page in range(1, max_pages + 1):
                print(f"Bintang {star} - Halaman {page}...", flush=True)
                slow_scroll_to(driver, 0.7)
                
                review_cards = []
                selectors = [
                    "[data-testid='divProductReview']",
                    ".css-1k41fl7",
                    ".css-72dthe",
                ]
                
                # Retry mencari elemen card hingga maksimal 15 detik
                for _ in range(5):
                    for sel in selectors:
                        cards = driver.find_elements(By.CSS_SELECTOR, sel)
                        if len(cards) >= 1:
                            review_cards = cards
                            break
                    if review_cards:
                        break
                    time.sleep(3)

                if not review_cards:
                    print(f"Card ulasan habis/gagal diload di hal {page} untuk bintang {star}.", flush=True)
                    break

                try:
                    expand_btns = driver.find_elements(
                        By.XPATH,
                        "//*[text()='Selengkapnya' or contains(text(), 'Tampilkan selengkapnya')]"
                    )
                    for btn in expand_btns:
                        try:
                            driver.execute_script("arguments[0].click();", btn)
                        except:
                            pass
                    if expand_btns:
                        time.sleep(1.5)
                except:
                    pass

                page_count = 0
                for card in review_cards:
                    try:
                        # Ambil Nama
                        name = ""
                        for sel in ["[data-testid='lblUserName']", "h3", "[class*='name']"]:
                            try:
                                t = card.find_element(By.CSS_SELECTOR, sel).text.strip()
                                if t:
                                    name = t
                                    break
                            except:
                                continue

                        # Ambil Rating
                        rating = star  # default ke bintang yang sedang aktif
                        try:
                            el = card.find_element(
                                By.CSS_SELECTOR,
                                "[aria-label*='bintang'], [aria-label*='star']"
                            )
                            aria = el.get_attribute("aria-label") or ""
                            nums = re.findall(r'\d+', aria)
                            if nums:
                                rating = int(nums[0])
                        except:
                            pass

                        # Ambil Komentar
                        comment = ""
                        try:
                            comment_el = card.find_element(
                                By.CSS_SELECTOR, "[data-testid='lblItemUlasan']"
                            )
                            comment = comment_el.text.strip()
                        except:
                            pass

                        if not comment:
                            try:
                                spans = card.find_elements(By.CSS_SELECTOR, "span, p")
                                for sp in spans:
                                    t = sp.text.strip()
                                    t = re.sub(
                                        r'(Selengkapnya|Tutup Ulasan)', '',
                                        t, flags=re.IGNORECASE
                                    ).strip()
                                    is_date = bool(re.search(
                                        r'(detik|menit|jam|hari|minggu|bulan|tahun)\s+lalu'
                                        r'|kemarin|hari ini',
                                        t, re.IGNORECASE
                                    ))
                                    if not is_date and len(t) > len(comment) and len(t) > 5:
                                        comment = t
                            except:
                                pass

                        # ---------------------------------------------------------
                        # PERBAIKAN: Hanya simpan jika 'comment' BENAR-BENAR ADA (tidak kosong)
                        # ---------------------------------------------------------
                        if comment and len(comment.strip()) > 0:
                            all_reviews.append({
                                "ProductName": product_name,
                                "Name": name,
                                "Rating": rating,
                                "Comment": comment,
                            })
                            page_count += 1
                    except:
                        continue

                print(f"+{page_count} ulasan di halaman ini.", flush=True)

                # Navigasi halaman selanjutnya
                clicked_next = False
                try:
                    next_btn = driver.find_element(
                        By.XPATH,
                        "//button[@aria-label='Laman berikutnya' or @aria-label='Next page']"
                    )
                    if next_btn.is_enabled():
                        driver.execute_script("arguments[0].click();", next_btn)
                        clicked_next = True
                        time.sleep(3)
                except:
                    pass

                if not clicked_next:
                    print(f"Halaman terakhir untuk bintang {star}.", flush=True)
                    break

            # Reset filter sebelum lanjut ke bintang berikutnya
            try:
                driver.execute_script("arguments[0].click();", filter_el)
                time.sleep(3)
                print(f"Filter Bintang {star} di-reset.", flush=True)
            except:
                print(f"Gagal reset filter bintang {star}.", flush=True)

    finally:
        driver.quit()

    return all_reviews


def main():
    try:
        if len(sys.argv) < 5:
            print(json.dumps({
                "status": "error",
                "message": "Argumen kurang. Diperlukan: url, product_name, max_pages, output_path"
            }), flush=True)
            sys.exit(1)

        url          = sys.argv[1]
        
        # ---------------------------------------------------------
        # PERBAIKAN: Potong nama produk agar maksimal 4 kata saja
        # ---------------------------------------------------------
        raw_name     = sys.argv[2]
        words        = raw_name.split()
        product_name = " ".join(words[:4]) 
        
        max_pages    = int(sys.argv[3]) if sys.argv[3].isdigit() else 50
        output_path  = sys.argv[4]

        print(f"Mulai scraping: {product_name}", flush=True)
        print(f"URL: {url}", flush=True)
        print(f"Max halaman per bintang: {max_pages}", flush=True)

        reviews = scrape_reviews(url, product_name, max_pages)

        if not reviews:
            print(json.dumps({
                "status": "error",
                "message": "Tidak ada ulasan yang berhasil di-scrape. Cek URL atau koneksi."
            }), flush=True)
            sys.exit(1)

        df = pd.DataFrame(reviews)
        df.to_csv(output_path, index=False, sep=",", encoding="utf-8-sig")

        print(f"Selesai. {len(reviews)} ulasan disimpan ke {output_path}", flush=True)

        print(json.dumps({
            "status":  "success",
            "total":   len(reviews),
            "output":  output_path,
        }), flush=True)
        
    except Exception as e:
        print(json.dumps({
            "status": "error",
            "message": f"Terjadi kesalahan di Python: {str(e)}"
        }), flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()