import asyncio
import logging
import json
import hashlib
import os
import aiohttp
from datetime import datetime, timezone
from typing import List, Dict, Set, Optional, Any
from pathlib import Path

# Try importing Anthropic, handle failure gracefully (though it should be installed)
try:
    from anthropic import AsyncAnthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

logger = logging.getLogger(__name__)

class TopicDiscoveryAgent:
    """
    Scans Prediction Markets (Polymarket) for new relevant topics.

    Features:
    - Dynamic discovery based on interest areas (tags, keywords)
    - LLM-based relevance assessment (Anthropic)
    - Position protection (Zombie Position Fix)
    - Budget Guard integration
    - Persistence of discovered topics
    """

    DISCOVERED_TOPICS_FILE = "data/discovered_topics.json"

    def __init__(self, config: dict, budget_guard=None):
        self.config = config
        self.sentinel_config = config.get('sentinels', {}).get('prediction_markets', {})
        self.discovery_config = self.sentinel_config.get('discovery_agent', {})
        self.interest_areas = self.sentinel_config.get('interest_areas', [])

        self.enabled = self.discovery_config.get('enabled', False)
        self.api_url = self.sentinel_config.get('providers', {}).get('polymarket', {}).get('api_url', "https://gamma-api.polymarket.com/events")

        # LLM Config
        self.llm_model = self.discovery_config.get('llm_model', "claude-3-haiku-20240307") # Fallback to known model if config has placeholder
        if "claude-haiku" in self.discovery_config.get('llm_model', ""):
             # Map config friendly name to actual model ID if needed, or trust config
             # The prompt suggested "claude-haiku-4-5-20251001" which doesn't exist yet, likely futuristic.
             # I will use the config value but fallback to a real model if it fails?
             # No, I should use what's in config or a safe default.
             # Let's use the config value.
             pass

        self.max_llm_calls = self.discovery_config.get('max_llm_calls_per_scan', 15)
        self._llm_calls_this_scan = 0

        # Filtering
        self.min_liquidity = self.discovery_config.get('min_liquidity_usd', 5000)
        self.min_volume = self.discovery_config.get('min_volume_usd', 5000)
        self.max_total_topics = self.discovery_config.get('max_total_topics', 12)

        # Budget Guard (Dependency Injection)
        self._budget_guard = budget_guard

        # Anthropic Client
        api_key = config.get('anthropic', {}).get('api_key')
        if not api_key or api_key == "LOADED_FROM_ENV":
            api_key = os.environ.get('ANTHROPIC_API_KEY')

        if HAS_ANTHROPIC and api_key:
            self.anthropic = AsyncAnthropic(api_key=api_key)
        else:
            self.anthropic = None
            if self.discovery_config.get('novel_market_llm_assessment', False):
                logger.warning("TopicDiscoveryAgent: Anthropic not available. LLM assessment disabled.")

    async def run_scan(self) -> Dict[str, Any]:
        """
        Main execution method.
        1. Discover candidates per interest area
        2. Deduplicate
        3. Filter & Sort
        4. Apply Caps (with Position Protection)
        5. Persist & Notify
        """
        if not self.enabled:
            return {'status': 'disabled', 'changes': {}, 'metadata': {}}

        logger.info("Starting Topic Discovery Scan...")
        self._llm_calls_this_scan = 0
        discovered_raw = []

        # 1. Discover Candidates
        for area in self.interest_areas:
            if not area.get('enabled', True):
                continue

            area_candidates = await self._scan_area(area)
            discovered_raw.extend(area_candidates)

        # 2. Global Deduplication (by slug)
        # Keep the one with highest relevance score, or merge?
        # Simple dedup: First one wins (or highest score)
        deduped = {}
        for item in discovered_raw:
            slug = item['slug']
            if slug not in deduped:
                deduped[slug] = item
            else:
                # Keep existing if score is higher, else replace
                if item['relevance_score'] > deduped[slug]['relevance_score']:
                    deduped[slug] = item

        candidates = list(deduped.values())

        # 3. Sort by Relevance then Liquidity
        candidates.sort(key=lambda x: (x['relevance_score'], x['liquidity']), reverse=True)

        # 4. Apply Global Cap (with Position Protection)
        # First, take top N
        final_topics = candidates[:self.max_total_topics]

        # === AMENDMENT A: POSITION PROTECTION ===
        protected_slugs = self._get_position_protected_slugs()
        if protected_slugs:
            current_slugs = {t['slug'] for t in final_topics}
            missing_protected = protected_slugs - current_slugs

            for slug in missing_protected:
                # Find in full candidate list
                protected_topic = next((c for c in candidates if c['slug'] == slug), None)
                if protected_topic:
                    final_topics.append(protected_topic)
                    logger.warning(f"POSITION PROTECTION: Forcing '{slug}' back into tracked topics.")
                else:
                    # Not in candidates (maybe delisted or liquidity drop below discovery threshold?)
                    # If we have it in TMS, we should probably construct a minimal record or warn.
                    # Warning is sufficient as per guide.
                    logger.warning(
                        f"POSITION PROTECTION: Active thesis for '{slug}' exists, but market not found in scan. "
                        f"Manual check recommended."
                    )

        # 5. Convert to Sentinel Format
        sentinel_topics = []
        for cand in final_topics:
            topic_config = self._convert_to_sentinel_config(cand)
            sentinel_topics.append(topic_config)

        # 6. Detect Changes
        changes = self._detect_changes(sentinel_topics)

        # 7. Persist (if changes or force update)
        if changes['has_changes'] and self.discovery_config.get('auto_apply', True):
            self._save_discovered_topics(sentinel_topics)
            if self.discovery_config.get('notify_on_change', True):
                self._notify_changes(changes)

        return {
            'status': 'success',
            'changes': changes,
            'metadata': {
                'topics_discovered': len(sentinel_topics),
                'llm_calls': self._llm_calls_this_scan
            }
        }

    async def _scan_area(self, area: Dict) -> List[Dict]:
        """Scan a specific interest area using defined methods."""
        candidates = []
        methods = area.get('discovery_methods', [])

        for method in methods:
            try:
                if method['type'] == 'tag_scan':
                    events = await self._fetch_events(tag_id=method.get('tag_id'), limit=method.get('limit', 10))
                elif method['type'] == 'query':
                    events = await self._fetch_events(query=method.get('q'), limit=20) # Limit query results
                else:
                    continue

                for event in events:
                    # === AMENDMENT D & B: Extract Best Market + Event ID ===
                    market_data = self._extract_market_data(event)
                    if not market_data:
                        continue

                    # Filter by Keywords (Layer 1)
                    relevance = self._score_relevance_keywords(market_data['title'], area)

                    # Filter by Exclude Keywords
                    if self._check_exclusions(market_data['title'], area):
                        continue

                    # LLM Assessment (Layer 2) - Only if enabled and ambiguous
                    # Simple heuristic: If relevance is low but not zero, or if novel market detection is on
                    if relevance > 0:
                        # Keyword match found
                        if relevance < area.get('min_relevance_score', 1):
                            # Try LLM boost if enabled
                            if self.discovery_config.get('novel_market_llm_assessment', False):
                                llm_score = await self._llm_assess_relevance(market_data, area)
                                if llm_score:
                                    relevance = max(relevance, llm_score)

                    if relevance >= area.get('min_relevance_score', 1):
                        market_data['relevance_score'] = relevance
                        market_data['interest_area'] = area['name']
                        market_data['commodity_impact_template'] = area.get('commodity_impact_template')
                        market_data['importance'] = area.get('importance', 'macro')
                        market_data['default_threshold_pct'] = area.get('default_threshold_pct', 8.0)

                        # === AMENDMENT B: Tag Generation with Event ID ===
                        market_data['tag'] = self._generate_tag(
                            market_data.get('event_id'),
                            market_data['slug'],
                            area['name']
                        )

                        candidates.append(market_data)

            except Exception as e:
                logger.error(f"Error scanning area '{area['name']}' method '{method}': {e}")

        return candidates

    async def _fetch_events(self, tag_id: int = None, query: str = None, limit: int = 10) -> List[Dict]:
        """Fetch events from Gamma API."""
        params = {
            'limit': limit,
            'closed': 'false',
            'active': 'true'
        }
        if tag_id:
            params['tag_id'] = tag_id
        if query:
            params['q'] = query

        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(self.api_url, params=params) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Gamma API error: {response.status}")
                        return []
        except Exception as e:
            logger.error(f"Gamma API fetch failed: {e}")
            return []

    def _extract_market_data(self, event: Dict[str, Any]) -> Optional[Dict]:
        """
        Extract normalized market data from a Gamma API event.

        AMENDMENT D: Select BEST market (highest liquidity) from event.
        AMENDMENT B: Capture event_id.
        """
        if not isinstance(event, dict):
            return None

        markets = event.get('markets', [])
        if not markets:
            return None

        # === AMENDMENT D: Select BEST market by liquidity ===
        best_market = None
        best_liquidity = -1

        for m in markets:
            if not isinstance(m, dict):
                continue
            try:
                liq = float(m.get('liquidity', 0) or 0)
            except (ValueError, TypeError):
                continue

            if liq > best_liquidity:
                best_liquidity = liq
                best_market = m

        if best_market is None:
            return None

        market = best_market

        try:
            liquidity = float(market.get('liquidity', 0) or 0)
            volume = float(market.get('volume', 0) or 0)
            # volume24hr is usually on event level
            volume_24h = float(event.get('volume24hr', 0) or 0)
        except (ValueError, TypeError):
            return None

        if liquidity < self.min_liquidity:
            return None
        if volume < self.min_volume:
            return None

        # Parse outcome price (usually first one for binary)
        try:
            prices_raw = market.get('outcomePrices', '[]')
            if isinstance(prices_raw, str):
                prices = json.loads(prices_raw)
            else:
                prices = prices_raw
            price = float(prices[0]) if prices else 0.0
        except (json.JSONDecodeError, ValueError, IndexError):
            return None

        return {
            'event_id': str(event.get('id', '')),  # AMENDMENT B: Stable identifier
            'slug': event.get('slug', ''),
            'title': event.get('title', ''),
            'price': price,
            'liquidity': liquidity,
            'volume': volume,
            'volume_24h': volume_24h,
            'market_count': len(markets),
        }

    def _generate_tag(self, event_id: str, slug: str, area_name: str) -> str:
        """
        Generate a short, deterministic tag.

        AMENDMENT B: Use event_id for hash stability if available.
        """
        area_prefix = ''.join(w[0].upper() for w in area_name.split()[:3])

        # Prefer event_id (stable)
        hash_source = event_id if event_id else slug
        source_hash = hashlib.md5(hash_source.encode()).hexdigest()[:4]

        return f"D_{area_prefix}_{source_hash}"

    def _score_relevance_keywords(self, title: str, area: Dict) -> int:
        """Score title against area keywords."""
        score = 0
        title_lower = title.lower()
        for kw in area.get('relevance_keywords', []):
            if kw.lower() in title_lower:
                score += 1
        return score

    def _check_exclusions(self, title: str, area: Dict) -> bool:
        """Check if title contains excluded keywords."""
        title_lower = title.lower()
        for kw in area.get('exclude_keywords', []):
            if kw.lower() in title_lower:
                return True
        return False

    async def _llm_assess_relevance(self, candidate: Dict, area: Dict) -> Optional[int]:
        """
        Ask LLM if this market is relevant to the interest area.

        AMENDMENT E: Check Budget Guard.
        """
        # AMENDMENT E: Budget Check
        if self._budget_guard and self._budget_guard.is_budget_hit:
            logger.info("TopicDiscovery: Skipping LLM assessment (budget hit)")
            return None

        if self._llm_calls_this_scan >= self.max_llm_calls:
            return None

        if not self.anthropic:
            return None

        self._llm_calls_this_scan += 1

        prompt = f"""
You are a relevance filter for a Coffee Trading Bot.
We are looking for prediction markets related to: {area['name']}
Relevance Keywords: {', '.join(area['relevance_keywords'])}

Market Title: "{candidate['title']}"

Question: Is this market substantively relevant to {area['name']}?
Answer ONLY with a JSON object: {{"relevant": true/false, "score": 0-5}}
"""
        try:
            response = await self._call_anthropic(prompt)

            # Guard against empty/whitespace responses
            if not response or not response.strip():
                logger.warning(
                    f"LLM assessment returned empty response for "
                    f"'{candidate.get('title', 'unknown')}'"
                )
                return None

            # Strip markdown code fences if present (handles single-line and multi-line)
            cleaned = response.strip()
            if "```" in cleaned:
                import re
                match = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
                if match:
                    cleaned = match.group(1).strip()

            data = json.loads(cleaned)
            if data.get('relevant'):
                return int(data.get('score', 0))
            return 0

        except json.JSONDecodeError as e:
            logger.warning(
                f"LLM assessment returned non-JSON for "
                f"'{candidate.get('title', 'unknown')}': "
                f"{response[:100] if response else '<empty>'}"
            )
            return None
        except Exception as e:
            logger.error(
                f"LLM assessment failed for "
                f"'{candidate.get('title', 'unknown')}': {e}"
            )
            return None

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        if not self.anthropic:
            raise ValueError("Anthropic client not initialized")

        try:
            message = await self.anthropic.messages.create(
                model=self.llm_model,
                max_tokens=100,
                temperature=0.0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return message.content[0].text
        except Exception as e:
            # Handle potential budget/rate limit errors
            logger.error(f"Anthropic API error: {e}")
            raise e

    def _get_position_protected_slugs(self) -> Set[str]:
        """
        AMENDMENT A: Zombie Position Fix.
        Query TMS for active theses that originate from Prediction Markets.
        """
        protected = set()
        try:
            from trading_bot.tms import TransactiveMemory
            tms = TransactiveMemory()

            # v3 FIX: Use existing method
            macro_theses = tms.get_active_theses_by_guardian("Macro")

            for thesis in macro_theses:
                rationale = thesis.get('primary_rationale', '').lower()
                is_pm_triggered = any(kw in rationale for kw in [
                    'prediction market', 'polymarket', 'probability shift',
                    'fed probability', 'election odds', 'tariff odds'
                ])

                if is_pm_triggered:
                    supporting = thesis.get('supporting_data', {})
                    slug = supporting.get('polymarket_slug', '')
                    if slug:
                        protected.add(slug)
                        logger.info(f"Position-protected: '{slug}'")

        except Exception as e:
            logger.warning(f"Failed to query TMS for position protection: {e}")

        return protected

    def _convert_to_sentinel_config(self, cand: Dict) -> Dict:
        """Convert discovery candidate to PredictionMarketSentinel topic config."""
        return {
            "query": cand['title'], # Using title as query for Sentinel is safe as it will resolve it
            "tag": cand['tag'],
            "display_name": cand['title'][:50], # Truncate for display
            "trigger_threshold_pct": cand['default_threshold_pct'],
            "importance": cand['importance'],
            "commodity_impact": cand['commodity_impact_template'],
            "min_relevance_score": 1, # Already filtered
            # Metadata for tracking
            "_discovery": {
                "interest_area": cand['interest_area'],
                "event_id": cand['event_id'],
                "slug": cand['slug'],
                "liquidity": cand['liquidity'],
                "discovered_at": datetime.now(timezone.utc).isoformat()
            }
        }

    def _save_discovered_topics(self, topics: List[Dict]):
        """Save topics to JSON file."""
        try:
            with open(self.DISCOVERED_TOPICS_FILE, 'w') as f:
                json.dump(topics, f, indent=2)
            logger.info(f"Saved {len(topics)} discovered topics to {self.DISCOVERED_TOPICS_FILE}")
        except Exception as e:
            logger.error(f"Failed to save discovered topics: {e}")

    def _detect_changes(self, new_topics: List[Dict]) -> Dict:
        """Detect changes from previous run."""
        changes = {'has_changes': False, 'added': [], 'removed': [], 'summary': 'No changes'}

        if not os.path.exists(self.DISCOVERED_TOPICS_FILE):
            changes['has_changes'] = True
            changes['added'] = [t['tag'] for t in new_topics]
            changes['added_display'] = [
                t.get('display_name', t.get('query', t['tag'])) for t in new_topics
            ]
            changes['summary'] = f"Initial scan: {len(new_topics)} topics found"
            return changes

        try:
            with open(self.DISCOVERED_TOPICS_FILE, 'r') as f:
                old_topics = json.load(f)

            old_tags = {t['tag'] for t in old_topics}
            new_tags = {t['tag'] for t in new_topics}

            added = new_tags - old_tags
            removed = old_tags - new_tags

            if added or removed:
                changes['has_changes'] = True
                changes['added'] = list(added)
                changes['removed'] = list(removed)

                # Build tag → display_name lookup for human-readable notifications
                new_display = {t['tag']: t.get('display_name', t.get('query', t['tag'])) for t in new_topics}
                old_display = {t['tag']: t.get('display_name', t.get('query', t['tag'])) for t in old_topics}

                changes['added_display'] = [new_display.get(tag, tag) for tag in added]
                changes['removed_display'] = [old_display.get(tag, tag) for tag in removed]

                changes['summary'] = f"Topics: +{len(added)} added, -{len(removed)} removed"

        except Exception as e:
            logger.warning(f"Error detecting changes: {e}")

        return changes

    def _notify_changes(self, changes: Dict):
        """Send Pushover notification with human-readable topic names."""
        from notifications import send_pushover_notification

        msg = f"<b>Topic Discovery Update</b>\n{changes['summary']}\n"
        if changes.get('added_display'):
            display_items = [name[:60] for name in changes['added_display'][:5]]
            msg += f"\n<b>Added:</b>\n" + "\n".join(f"• {name}" for name in display_items)
        elif changes.get('added'):
            # Fallback to tags if display names unavailable
            msg += f"\n<b>Added:</b> {', '.join(changes['added'][:5])}"

        if changes.get('removed_display'):
            display_items = [name[:60] for name in changes['removed_display'][:5]]
            msg += f"\n<b>Removed:</b>\n" + "\n".join(f"• {name}" for name in display_items)
        elif changes.get('removed'):
            msg += f"\n<b>Removed:</b> {', '.join(changes['removed'][:5])}"

        send_pushover_notification(self.config.get('notifications', {}), "Topic Discovery", msg)
