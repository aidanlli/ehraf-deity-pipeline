"""Scrape paragraph-source metadata for every eHRAF culture via Selenium.

Step 01 of the scraping pipeline (~3 h). For each culture in the master
list, runs a subject-filtered search on ehrafworldcultures.yale.edu and
exports the result grid to a CSV in the browser's download folder. Move
the downloaded files into ``data/intermediate/ehraf_exports/`` afterwards.
"""

import sys
import time
from pathlib import Path
from urllib.parse import quote

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import CULTURE_SUMMARY_CSV


def search_ehraf(culture, subjects):
    subjects_query = " OR ".join([f'"{subject}"' for subject in subjects])
    search_url = f"https://ehrafworldcultures.yale.edu/search?q=cultures:%22{quote(culture)}%22%20AND%20subjects:({quote(subjects_query)})"

    # Setup Chrome options
    options = Options()
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-infobars")

    # optional arg: removes chrome window from popping up. You may need to comment this out later - see the README for more details.
    options.add_argument("--headless")

    # Initialize WebDriver
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    driver.get(search_url)
    wait = WebDriverWait(driver, 10)

    try:
        # Click region button (with retry mechanism)
        for attempt in range(3):  # Try 3 times
            try:
                print(f"Attempt {attempt + 1}: Clicking the region button...")
                region_button = wait.until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//div[contains(@class, 'trad-overview__result')]/h4/button")
                    )
                )
                region_button.click()
                time.sleep(3)  # Give time for the page to update
                break
            except Exception as e:
                print(f"Retry {attempt + 1}: {e}")
                continue

        # Wait until the region section expands
        wait.until(
            EC.presence_of_element_located(
                (By.XPATH, "//div[contains(@class, 'trad-overview__result--open')]")
            )
        )

        # Click culture link
        for attempt in range(3):
            try:
                print(f"Attempt {attempt + 1}: Clicking culture link...")
                culture_link = wait.until(
                    EC.element_to_be_clickable(
                        (By.XPATH, f'//a[contains(translate(., "’‘", "\'\'"), "{culture}")]')
                    )
                )

                culture_link.click()
                time.sleep(3)
                break
            except Exception as e:
                print(f"Retry {attempt + 1}: {e}")
                continue

        # Click the checkbox for selection
        for attempt in range(3):
            try:
                print(f"Attempt {attempt + 1}: Clicking checkbox...")
                checkbox = wait.until(
                    EC.presence_of_element_located((By.XPATH, "//input[@type='checkbox']"))
                )
                driver.execute_script(
                    "arguments[0].click();", checkbox
                )  # Ensure JavaScript handles the click
                time.sleep(1)
                break
            except Exception as e:
                print(f"Retry {attempt + 1}: {e}")
                continue

        # Click export/download button
        for attempt in range(3):
            try:
                print(f"Attempt {attempt + 1}: Clicking download button...")
                download_button = wait.until(
                    EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'get_app')]"))
                )
                download_button.click()
                time.sleep(1)
                break
            except Exception as e:
                print(f"Retry {attempt + 1}: {e}")
                continue

        # Choose export option
        for attempt in range(3):
            try:
                print(f"Attempt {attempt + 1}: Selecting export option...")
                export_option = wait.until(
                    EC.element_to_be_clickable(
                        (By.XPATH, "//li[contains(text(), 'Export to a CSV File')]")
                    )
                )
                export_option.click()
                time.sleep(3)
                print("✅ Exported data to CSV successfully.")
                break
            except Exception as e:
                print(f"Retry {attempt + 1}: {e}")
                continue

    except Exception as e:
        print(f"⚠️ Error: {e}")
    finally:
        driver.quit()


# Example usage
# search_ehraf("Rwala Bedouin", ["spirits and gods", "traditional history", "mythology"])
df = pd.read_csv(CULTURE_SUMMARY_CSV)
cultures = df["EHRAF WORLD CULTURES NAME"].tolist() + ["Chiricahua Apache"]
# uncomment the following cultures = line after the macro, comment out the previous cultures = section.
# You may have to manually help the macro navigate to the download page.
# cultures = ['Yoruba', 'Iroquois', 'Ainu']

subjects = [
    "cult of the dead",
    "general character of religion",
    "cosmology",
    "mythology",
    "animism",
    "eschatology",
    "spirits and gods",
    "sacred objects and places",
    "theological systems",
    "revelation and divination",
    "luck and chances",
]
for culture in cultures:
    print(f"🔍 Searching for culture: {culture}")
    search_ehraf(culture, subjects)

size_issue_list = ["Dogon", "Navajo", "Ifugao", "Hopi", "Zulu"]
