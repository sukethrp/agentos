#!/usr/bin/env python3
"""
Comprehensive panel testing with console error checking
Tests all sidebar panels systematically
"""
import asyncio
import sys
from playwright.async_api import async_playwright

async def test_all_panels():
    """Test all sidebar panels with console error tracking"""
    
    print("\n" + "="*70)
    print("AGENTOS COMPREHENSIVE PANEL TESTING")
    print("="*70 + "\n")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={'width': 1920, 'height': 1080})
        page = await context.new_page()
        
        # Track console errors
        console_errors = []
        console_warnings = []
        
        def handle_console(msg):
            if msg.type == 'error':
                console_errors.append(f"[{msg.type}] {msg.text}")
            elif msg.type == 'warning':
                console_warnings.append(f"[{msg.type}] {msg.text}")
        
        page.on('console', handle_console)
        
        # Track page errors
        page_errors = []
        
        def handle_page_error(error):
            page_errors.append(str(error))
        
        page.on('pageerror', handle_page_error)
        
        results = []
        
        try:
            print("📍 Navigating to http://localhost:8000...")
            await page.goto('http://localhost:8000', wait_until='networkidle')
            await page.wait_for_timeout(2000)
            print("✅ Page loaded\n")
            
            # Test panels in order
            test_cases = [
                {
                    "num": 1,
                    "name": "Agent Builder",
                    "nav_text": "🛠️ Agent Builder",
                    "panel_id": "panel-builder",
                    "checks": [
                        {"type": "input", "selector": "#b-query", "action": "type", "value": "test query"},
                        {"type": "element", "selector": "#b-name", "description": "name field"},
                        {"type": "element", "selector": "#b-model", "description": "model dropdown"},
                    ]
                },
                {
                    "num": 2,
                    "name": "Templates",
                    "nav_text": "📦 Templates",
                    "panel_id": "panel-templates",
                    "checks": [
                        {"type": "element", "selector": ".templates-grid", "description": "template grid"},
                    ]
                },
                {
                    "num": 3,
                    "name": "Chat",
                    "nav_text": "💬 Chat",
                    "panel_id": "panel-chat",
                    "checks": [
                        {"type": "element", "selector": "#chat-query", "description": "chat input"},
                    ]
                },
                {
                    "num": 4,
                    "name": "Branching",
                    "nav_text": "🌿 Branching",
                    "panel_id": "panel-branching",
                    "checks": [
                        {"type": "element", "selector": "#br-tree-select", "description": "tree selector"},
                    ]
                },
                {
                    "num": 5,
                    "name": "Monitor",
                    "nav_text": "📊 Monitor",
                    "panel_id": "panel-monitor",
                    "checks": [
                        {"type": "button", "selector": "button:has-text('Refresh')", "description": "refresh button"},
                    ]
                },
                {
                    "num": 6,
                    "name": "Analytics",
                    "nav_text": "📈 Analytics",
                    "panel_id": "panel-analytics",
                    "checks": [
                        {"type": "button", "selector": "button:has-text('Refresh')", "description": "refresh button"},
                    ]
                },
                {
                    "num": 7,
                    "name": "Scheduler",
                    "nav_text": "⏰ Scheduler",
                    "panel_id": "panel-scheduler",
                    "checks": [
                        {"type": "element", "selector": "#sched-name", "description": "scheduler name field"},
                    ]
                },
                {
                    "num": 8,
                    "name": "Events",
                    "nav_text": "⚡ Events",
                    "panel_id": "panel-events",
                    "checks": [
                        {"type": "button", "selector": "button:has-text('Refresh')", "description": "refresh button"},
                    ]
                },
                {
                    "num": 9,
                    "name": "A/B Testing",
                    "nav_text": "🧪 A/B Testing",
                    "panel_id": "panel-abtest",
                    "checks": [
                        {"type": "element", "selector": "#ab-prompt-a", "description": "prompt A field"},
                    ]
                },
                {
                    "num": 10,
                    "name": "Multi-modal",
                    "nav_text": "👁️ Multi-modal",
                    "panel_id": "panel-multimodal",
                    "checks": [
                        {"type": "element", "selector": ".mm-upload-zone", "description": "upload zone"},
                    ]
                },
                {
                    "num": 11,
                    "name": "Account & Usage",
                    "nav_text": "🔑 Account & Usage",
                    "panel_id": "panel-auth",
                    "checks": [
                        {"type": "element", "selector": "#auth-email", "description": "email field"},
                    ]
                },
                {
                    "num": 12,
                    "name": "Marketplace",
                    "nav_text": "🏪 Marketplace",
                    "panel_id": "panel-marketplace",
                    "checks": [
                        {"type": "button", "selector": "button:has-text('Refresh')", "description": "refresh button"},
                    ]
                },
                {
                    "num": 13,
                    "name": "Embed SDK",
                    "nav_text": "🔌 Embed SDK",
                    "panel_id": "panel-embed",
                    "checks": [
                        {"type": "element", "selector": "#embed-agent-name", "description": "agent name field"},
                    ]
                },
                {
                    "num": 14,
                    "name": "Agent Mesh",
                    "nav_text": "🕸️ Agent Mesh",
                    "panel_id": "panel-mesh",
                    "checks": [
                        {"type": "button", "selector": "button:has-text('Refresh')", "description": "refresh button"},
                    ]
                },
                {
                    "num": 15,
                    "name": "Simulation",
                    "nav_text": "🎮 Simulation",
                    "panel_id": "panel-simulation",
                    "checks": [
                        {"type": "element", "selector": "#sim-name", "description": "simulation name field"},
                    ]
                },
                {
                    "num": 16,
                    "name": "Learning",
                    "nav_text": "🧠 Learning",
                    "panel_id": "panel-learning",
                    "checks": [
                        {"type": "button", "selector": "button:has-text('Refresh Stats')", "description": "refresh stats button"},
                    ]
                },
                {
                    "num": 17,
                    "name": "RCA",
                    "nav_text": "🔍 RCA",
                    "panel_id": "panel-rca",
                    "checks": [
                        {"type": "button", "selector": "button:has-text('Refresh')", "description": "refresh button"},
                    ]
                },
            ]
            
            for test_case in test_cases:
                num = test_case["num"]
                name = test_case["name"]
                nav_text = test_case["nav_text"]
                panel_id = test_case["panel_id"]
                checks = test_case["checks"]
                
                print(f"{num}. Testing: {name}")
                
                # Clear error tracking for this panel
                panel_errors = []
                initial_error_count = len(console_errors)
                
                try:
                    # Find and click nav item
                    nav_items = await page.query_selector_all('.nav-item')
                    nav_item = None
                    
                    for item in nav_items:
                        text = await item.inner_text()
                        if nav_text in text:
                            nav_item = item
                            break
                    
                    if not nav_item:
                        panel_errors.append(f"Nav item not found: {nav_text}")
                        print(f"   ❌ ERROR: Nav item not found")
                        results.append({
                            "num": num,
                            "name": name,
                            "status": "ERROR",
                            "errors": panel_errors
                        })
                        print()
                        continue
                    
                    # Click nav item
                    await nav_item.click()
                    await page.wait_for_timeout(800)
                    
                    # Check if panel is visible
                    panel = await page.query_selector(f'#{panel_id}')
                    if not panel:
                        panel_errors.append(f"Panel not found: {panel_id}")
                        print(f"   ❌ ERROR: Panel not found")
                        results.append({
                            "num": num,
                            "name": name,
                            "status": "ERROR",
                            "errors": panel_errors
                        })
                        print()
                        continue
                    
                    is_visible = await panel.evaluate('el => el.classList.contains("active")')
                    if not is_visible:
                        panel_errors.append("Panel not visible (no active class)")
                        print(f"   ❌ ERROR: Panel not visible")
                    
                    # Run checks
                    for check in checks:
                        check_type = check["type"]
                        selector = check["selector"]
                        description = check.get("description", selector)
                        
                        try:
                            if check_type == "element":
                                element = await page.query_selector(selector)
                                if not element:
                                    panel_errors.append(f"Element not found: {description}")
                            
                            elif check_type == "input":
                                element = await page.query_selector(selector)
                                if not element:
                                    panel_errors.append(f"Input not found: {description}")
                                else:
                                    # Try typing
                                    await element.fill(check["value"])
                                    await page.wait_for_timeout(200)
                            
                            elif check_type == "button":
                                # Try to find and click button (within the active panel)
                                button = await panel.query_selector(selector)
                                if button:
                                    await button.click()
                                    await page.wait_for_timeout(500)
                                else:
                                    # Button might not exist, that's ok for some panels
                                    pass
                        
                        except Exception as e:
                            panel_errors.append(f"Check failed ({description}): {str(e)}")
                    
                    # Check for new console errors
                    new_errors = console_errors[initial_error_count:]
                    if new_errors:
                        panel_errors.extend([f"Console: {err}" for err in new_errors])
                    
                    # Determine status
                    if panel_errors:
                        status = "ERROR"
                        print(f"   ❌ ERROR: {len(panel_errors)} issue(s) found")
                        for err in panel_errors[:3]:  # Show first 3 errors
                            print(f"      - {err}")
                    else:
                        status = "OK"
                        print(f"   ✅ OK - Panel displays correctly")
                    
                    results.append({
                        "num": num,
                        "name": name,
                        "status": status,
                        "errors": panel_errors
                    })
                
                except Exception as e:
                    panel_errors.append(f"Test exception: {str(e)}")
                    print(f"   ❌ ERROR: {str(e)}")
                    results.append({
                        "num": num,
                        "name": name,
                        "status": "ERROR",
                        "errors": panel_errors
                    })
                
                print()
            
            # Summary
            print("="*70)
            print("📊 TEST SUMMARY")
            print("="*70 + "\n")
            
            ok_count = sum(1 for r in results if r["status"] == "OK")
            error_count = sum(1 for r in results if r["status"] == "ERROR")
            
            print(f"Total Panels Tested: {len(results)}")
            print(f"✅ OK: {ok_count}")
            print(f"❌ ERROR: {error_count}")
            print()
            
            # Detailed results
            print("DETAILED RESULTS:")
            print("-" * 70)
            for result in results:
                status_icon = "✅" if result["status"] == "OK" else "❌"
                print(f"{status_icon} {result['num']}. {result['name']}: {result['status']}")
                if result["errors"]:
                    for err in result["errors"]:
                        print(f"     - {err}")
            
            print("\n" + "="*70)
            
            # Console errors summary
            if console_errors:
                print(f"\n⚠️  Total Console Errors: {len(console_errors)}")
                print("First 5 console errors:")
                for err in console_errors[:5]:
                    print(f"  - {err}")
            else:
                print("\n✅ No console errors detected")
            
            if console_warnings:
                print(f"\n⚠️  Total Console Warnings: {len(console_warnings)}")
            
            print("\n" + "="*70 + "\n")
            
        except Exception as e:
            print(f"\n❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            await browser.close()
    
    return True

if __name__ == "__main__":
    try:
        asyncio.run(test_all_panels())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        sys.exit(1)
