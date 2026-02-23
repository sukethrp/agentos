#!/usr/bin/env python3
"""Quick smoke test - click all panels and check for errors"""
import asyncio
from playwright.async_api import async_playwright

async def smoke_test():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page(viewport={'width': 1920, 'height': 1080})
        
        console_errors = []
        page.on('console', lambda msg: console_errors.append(msg.text) if msg.type == 'error' and 'favicon' not in msg.text else None)
        
        print("🔥 QUICK SMOKE TEST\n")
        
        await page.goto('http://localhost:8000')
        await page.wait_for_function('typeof showPanel === "function"', timeout=10000)
        
        panels = [
            "Agent Builder", "Templates", "Chat", "Branching", "Monitor", 
            "Analytics", "Scheduler", "Events", "A/B Testing", "Multi-modal",
            "Account & Usage", "Marketplace", "Embed SDK", "Agent Mesh", 
            "Simulation", "Learning", "RCA"
        ]
        
        panel_ids = [
            "builder", "templates", "chat", "branching", "monitor",
            "analytics", "scheduler", "events", "abtest", "multimodal",
            "auth", "marketplace", "embed", "mesh",
            "simulation", "learning", "observability"
        ]
        
        issues = []
        
        for name, pid in zip(panels, panel_ids):
            await page.evaluate(f"showPanel('{pid}')")
            await page.wait_for_timeout(300)
            
            visible = await page.evaluate(f"""() => {{
                const p = document.getElementById('panel-{pid}');
                return p && p.classList.contains('active') && p.offsetParent !== null;
            }}""")
            
            if not visible:
                issues.append(f"{name}: NOT VISIBLE")
                print(f"❌ {name}")
            else:
                print(f"✅ {name}")
        
        # Test RCA refresh button
        await page.evaluate("showPanel('observability')")
        await page.wait_for_timeout(300)
        try:
            btn = await page.query_selector("#panel-observability button:has-text('Refresh')")
            if btn:
                await btn.click()
                await page.wait_for_timeout(500)
                print("✅ RCA Refresh button clicked")
        except:
            print("⚠️  RCA Refresh button not found")
        
        print(f"\n📊 RESULTS:")
        print(f"Console Errors: {len(console_errors)}")
        print(f"Panels Not Displaying: {len(issues)}")
        
        if console_errors:
            print(f"\nErrors:")
            for err in console_errors[:3]:
                print(f"  - {err[:80]}")
        
        if issues:
            print(f"\nIssues:")
            for issue in issues:
                print(f"  - {issue}")
        
        if not console_errors and not issues:
            print("🎉 ALL PANELS OK - NO ERRORS")
        
        await browser.close()

asyncio.run(smoke_test())
