"""AtlasLobby — SEO platform overview narrative.

Returns a markdown overview of what FlowCentral SEO has to offer.
Atlas calls this tool and relays the info to new visitors in his own words.
"""

import atlantis
import logging

logger = logging.getLogger("mcp_server")


@visible
async def get_overview():
    """
    Returns a markdown overview of the FlowCentral SEO platform
    for Atlas to relay to new visitors.
    """
    logger.info("AtlasLobby get_overview called")

    return """## FlowCentral SEO Tools

We're building a full-suite SEO platform — 34 professional analysis tools
with rich HTML reports and downloadable PDF exports. Pay-per-use pricing,
no subscriptions.

### What's here (and coming)

**Site Analysis** — Page speed & Core Web Vitals, on-page SEO audits,
technical audits (SSL, redirects, headers), full-site crawling up to 500 pages,
email deliverability (DNS/SPF/DKIM/DMARC).

**Keyword Research** — Search volume, CPC, difficulty, keyword gap analysis,
keyword clustering, topic research, free keyword ideation.

**Backlinks** — Full backlink profiles, backlink gap analysis, broken backlink
finder, toxic backlink audit with disavow file generation.

**Competitive Intelligence** — Domain overviews with traffic estimates,
competitor content discovery, content explorer, SERP feature tracking.

**Content** — TF-IDF content optimization vs top Google competitors,
SEO writing assistant, internal link suggestions.

**Monitoring** — Rank tracking, backlink change alerts, site health checks,
brand mention monitoring, uptime & SSL monitoring.

**Advanced** — Local SEO & Google Business Profile, PPC/ad research,
social media tracking, AI visibility (how LLMs talk about your brand).

### What's live right now

🟢 **Page Speed** — Full Google Lighthouse audit. Runs both mobile AND desktop
in parallel. Gets you Core Web Vitals (LCP, FCP, CLS, TBT), performance scores,
resource breakdowns, third-party impact analysis, and prioritized fix
instructions with code examples. Free with a Google API key.

*More tools are being rolled out — page_speed is the one to try right now.*

### Pricing philosophy

No subscriptions. You pay per report. A freelancer checking their own site
might spend $4-5/month. An agency running 5 client sites: ~$60/month.
Compare that to Ahrefs at $129/month or SEMrush at $140/month.
"""
