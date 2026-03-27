import asyncio
from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        
        page.on("console", lambda msg: print(f"Console: {msg.text}"))
        page.on("pageerror", lambda err: print(f"PageError: {err}"))
        
        await page.goto("http://127.0.0.1:5000/")
        await asyncio.sleep(2)
        await browser.close()

asyncio.run(main())
