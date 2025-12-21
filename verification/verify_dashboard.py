from playwright.sync_api import sync_playwright
import time

def verify():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            print("Navigating to http://localhost:8503...")
            page.goto("http://localhost:8503", timeout=60000)

            print("Waiting for Title 'Coffee Bot Mission Control'...")
            page.wait_for_selector("h1", timeout=60000)

            print("Page loaded. Looking for 'Council Scorecard' tab...")
            # Use get_by_role to target the tab button specifically
            page.get_by_role("tab", name="Council Scorecard").click()
            time.sleep(5)

            print("Taking screenshot...")
            page.screenshot(path="verification/dashboard_success.png", full_page=True)

            content = page.content()
            if "Agent Accuracy Leaderboard" in content:
                print("VERIFIED: Leaderboard found.")
            else:
                print("WARNING: Leaderboard text not found in source.")

        except Exception as e:
            print(f"Error: {e}")
            page.screenshot(path="verification/dashboard_error_8503.png")
        finally:
            browser.close()

if __name__ == "__main__":
    verify()
