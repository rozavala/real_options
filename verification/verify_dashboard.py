
from playwright.sync_api import sync_playwright, expect
import time

def verify_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={'width': 1280, 'height': 1024})
        page = context.new_page()

        print("Navigating to Cockpit...")
        try:
            page.goto("http://localhost:8501/Cockpit", timeout=15000)
            page.wait_for_load_state("networkidle")
            time.sleep(3)

            # Click the expander to reveal X Sentiment Sensor
            print("Opening Sentinel Details expander...")
            # Streamlit expanders are usually summaries.
            page.get_by_text("🔍 Sentinel Details").click()
            time.sleep(1)

            print("Checking Cockpit content...")
            # Now it should be visible
            expect(page.get_by_text("X Sentiment Sensor")).to_be_visible(timeout=5000)

            # Check for Market Regime (this is outside expander)
            expect(page.get_by_text("Current Market Regime")).to_be_visible(timeout=5000)

            page.screenshot(path="verification/cockpit.png")
            print("Captured verification/cockpit.png")

        except Exception as e:
            print(f"Cockpit verification failed: {e}")
            page.screenshot(path="verification/cockpit_error.png")

        print("Navigating to Financials...")
        try:
            page.goto("http://localhost:8501/Financials", timeout=15000)
            page.wait_for_load_state("networkidle")
            time.sleep(3)

            print("Checking Financials content...")
            expect(page.get_by_text("Synthetic Equity Curve")).to_be_visible(timeout=5000)

            page.screenshot(path="verification/financials.png")
            print("Captured verification/financials.png")

        except Exception as e:
            print(f"Financials verification failed: {e}")
            page.screenshot(path="verification/financials_error.png")

        browser.close()

if __name__ == "__main__":
    verify_dashboard()
