"""Task 11 fixture generator (throwaway spike, NOT a product feature).

Writes the bake-off eval fixture set into ``<repo>/data/eval/``:

- ``text/human/human_01..20.txt`` — 20 public-domain human-written excerpts
  (Project Gutenberg, pre-1930 books, whitespace-collapsed single paragraphs),
  each with per-book provenance below;
- ``text/ai/ai_01..20.txt`` — 20 agent-composed ChatGPT-style outputs
  (written by the coding agent for this task; NOT collected ChatGPT traffic),
  ChatGPT-typical register (polite opener, structure, closing offer);
- ``images/real/real_01..10.jpg`` — 9 NASA-image-library photographs (public
  domain) + 1 model-author demo photograph (Smaug Space example);
- ``images/ai/ai_01..10.jpg`` — 9 Wikimedia Commons AI-generated images
  (per-tool categories, CC0/PD/CC-BY-SA; fetched via the CDN-cached 1280px
  thumbnail endpoints) + 1 model-author demo AI image (Smaug Space example);
- ``MANIFEST.md`` — provenance + sha256 for every written file.

Image files are downloaded from the sources in ``IMAGE_SOURCES`` (network
needed), EXIF-transposed, alpha-composited onto white, converted to RGB,
downscaled to a 1280 px long side and re-encoded as JPEG q88 — so the same
source URL reproduces the same committed bytes.

Text content is embedded below (no network needed for text).

Usage::

    cd backend && .venv/bin/python tests/fixtures/make_fixtures.py
    # flags: --text-only / --images-only / --max-side N / --quality N

This file is deliberately NOT a pytest module (name does not match
``test_*.py``), so the offline suite stays fully offline; nothing here runs at
import time.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = REPO_ROOT / "data" / "eval"
USER_AGENT = "ai-content-detector-bakeoff-fixture-generator/1.0 (task-11 spike)"
SMALL_PX = 500  # reject sources smaller than this on the short side


def _flatten(source: str) -> str:
    """Rebuild fixture text from source-formatted lines (<=100 cols for ruff).

    Paragraph blocks (separated by blank lines) are joined with single spaces
    inside a paragraph and blank lines between paragraphs; leading/trailing
    whitespace per line is dropped. Gutenberg excerpts therefore reproduce the
    extracted whitespace-collapsed paragraph exactly.
    """
    paragraphs = [block.strip() for block in source.strip().split("\n\n")]
    joined = (" ".join(line.strip() for line in block.splitlines()) for block in paragraphs)
    return "\n\n".join(joined)


# --- human text: 20 public-domain excerpts (Project Gutenberg, pre-1930) ----
# Provenance keys: (filename, title, author, year, gutenberg id, excerpt text).
# Excerpts are single qualifying paragraphs (280-620 chars after whitespace
# collapsing), hand-picked from the cited ebook's plain-text edition.

TEXT_HUMAN: list[dict[str, Any]] = [
    {
        "file": "human_01.txt",
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "year": 1813,
        "gutenberg_id": 1342,
        "text": """
            “What can be the meaning of that emphatic exclamation?” cried he. “Do you consider the
            forms of introduction, and the stress that is laid on them, as nonsense? I cannot quite
            agree with you _there_. What say you, Mary? For you are a young lady of deep reflection,
            I know, and read great books, and make extracts.”
        """,
    },
    {
        "file": "human_02.txt",
        "title": "Pride and Prejudice",
        "author": "Jane Austen",
        "year": 1813,
        "gutenberg_id": 1342,
        "text": """
            “I hope,” said she, as they were walking together in the shrubbery the next day, “you
            will give your mother-in-law a few hints, when this desirable event takes place, as to
            the advantage of holding her tongue; and if you can compass it, to cure the younger
            girls of running after the officers. And, if I may mention so delicate a subject,
            endeavour to check that little something, bordering on conceit and impertinence, which
            your lady possesses.”
        """,
    },
    {
        "file": "human_03.txt",
        "title": "Moby-Dick",
        "author": "Herman Melville",
        "year": 1851,
        "gutenberg_id": 2701,
        "text": """
            “We set sail from the Elbe, wind N.E. in the ship called The Jonas-in-the-Whale.... Some
            say the whale can’t open his mouth, but that is a fable.... They frequently climb up the
            masts to see whether they can see a whale, for the first discoverer has a ducat for his
            pains.... I was told of a whale taken near Shetland, that had above a barrel of herrings
            in his belly.... One of our harpooneers told me that he caught once a whale in
            Spitzbergen that was white all over.” —_A Voyage to Greenland, A.D._ 1671. _Harris Coll_.
        """,
    },
    {
        "file": "human_04.txt",
        "title": "Moby-Dick",
        "author": "Herman Melville",
        "year": 1851,
        "gutenberg_id": 2701,
        "text": """
            At last some four or five of us were summoned to our meal in an adjoining room. It was
            cold as Iceland—no fire at all—the landlord said he couldn’t afford it. Nothing but two
            dismal tallow candles, each in a winding sheet. We were fain to button up our monkey
            jackets, and hold to our lips cups of scalding tea with our half frozen fingers. But the
            fare was of the most substantial kind—not only meat and potatoes, but dumplings; good
            heavens! dumplings for supper! One young fellow in a green box coat, addressed himself
            to these dumplings in a most direful manner.
        """,
    },
    {
        "file": "human_05.txt",
        "title": "Frankenstein",
        "author": "Mary Shelley",
        "year": 1818,
        "gutenberg_id": 84,
        "text": """
            I shall depart for the latter town in a fortnight or three weeks; and my intention is to
            hire a ship there, which can easily be done by paying the insurance for the owner, and
            to engage as many sailors as I think necessary among those who are accustomed to the
            whale-fishing. I do not intend to sail until the month of June; and when shall I return?
            Ah, dear sister, how can I answer this question? If I succeed, many, many months,
            perhaps years, will pass before you and I may meet. If I fail, you will see me again
            soon, or never.
        """,
    },
    {
        "file": "human_06.txt",
        "title": "Frankenstein",
        "author": "Mary Shelley",
        "year": 1818,
        "gutenberg_id": 84,
        "text": """
            Henry rejoiced in my gaiety, and sincerely sympathised in my feelings: he exerted himself
            to amuse me, while he expressed the sensations that filled his soul. The resources of
            his mind on this occasion were truly astonishing: his conversation was full of
            imagination; and very often, in imitation of the Persian and Arabic writers, he invented
            tales of wonderful fancy and passion. At other times he repeated my favourite poems, or
            drew me out into arguments, which he supported with great ingenuity.
        """,
    },
    {
        "file": "human_07.txt",
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "year": 1865,
        "gutenberg_id": 11,
        "text": """
            There seemed to be no use in waiting by the little door, so she went back to the table,
            half hoping she might find another key on it, or at any rate a book of rules for shutting
            people up like telescopes: this time she found a little bottle on it, (“which certainly
            was not here before,” said Alice,) and round the neck of the bottle was a paper label,
            with the words “DRINK ME,” beautifully printed on it in large letters.
        """,
    },
    {
        "file": "human_08.txt",
        "title": "Alice's Adventures in Wonderland",
        "author": "Lewis Carroll",
        "year": 1865,
        "gutenberg_id": 11,
        "text": """
            “Here! you may nurse it a bit, if you like!” the Duchess said to Alice, flinging the baby
            at her as she spoke. “I must go and get ready to play croquet with the Queen,” and she
            hurried out of the room. The cook threw a frying-pan after her as she went out, but it
            just missed her.
        """,
    },
    {
        "file": "human_09.txt",
        "title": "The Adventures of Sherlock Holmes",
        "author": "Arthur Conan Doyle",
        "year": 1892,
        "gutenberg_id": 1661,
        "text": """
            “There will call upon you to-night, at a quarter to eight o’clock,” it said, “a gentleman
            who desires to consult you upon a matter of the very deepest moment. Your recent
            services to one of the royal houses of Europe have shown that you are one who may safely
            be trusted with matters which are of an importance which can hardly be exaggerated. This
            account of you we have from all quarters received. Be in your chamber then at that hour,
            and do not take it amiss if your visitor wear a mask.”
        """,
    },
    {
        "file": "human_10.txt",
        "title": "The Adventures of Sherlock Holmes",
        "author": "Arthur Conan Doyle",
        "year": 1892,
        "gutenberg_id": 1661,
        "text": """
            It was a quarter-past nine when I started from home and made my way across the Park, and
            so through Oxford Street to Baker Street. Two hansoms were standing at the door, and as
            I entered the passage I heard the sound of voices from above. On entering his room, I
            found Holmes in animated conversation with two men, one of whom I recognised as Peter
            Jones, the official police agent, while the other was a long, thin, sad-faced man, with
            a very shiny hat and oppressively respectable frock-coat.
        """,
    },
    {
        "file": "human_11.txt",
        "title": "Dracula",
        "author": "Bram Stoker",
        "year": 1897,
        "gutenberg_id": 345,
        "text": """
            “That is why, I suppose, you wished him to go on to Bukovina. You cannot deceive me, my
            friend; I know too much, and my horses are swift.” As he spoke he smiled, and the
            lamplight fell on a hard-looking mouth, with very red lips and sharp-looking teeth, as
            white as ivory. One of my companions whispered to another the line from Burger’s
            “Lenore”:--
        """,
    },
    {
        "file": "human_12.txt",
        "title": "Dracula",
        "author": "Bram Stoker",
        "year": 1897,
        "gutenberg_id": 345,
        "text": """
            It is nice at high water; but when the tide is out it shoals away to nothing, and there
            is merely the stream of the Esk, running between banks of sand, with rocks here and
            there. Outside the harbour on this side there rises for about half a mile a great reef,
            the sharp edge of which runs straight out from behind the south lighthouse. At the end
            of it is a buoy with a bell, which swings in bad weather, and sends in a mournful sound
            on the wind. They have a legend here that when a ship is lost bells are heard out at
            sea. I must ask the old man about this; he is coming this way....
        """,
    },
    {
        "file": "human_13.txt",
        "title": "The Time Machine",
        "author": "H. G. Wells",
        "year": 1895,
        "gutenberg_id": 35,
        "text": """
            “My dear sir, that is just where you are wrong. That is just where the whole world has
            gone wrong. We are always getting away from the present moment. Our mental existences,
            which are immaterial and have no dimensions, are passing along the Time-Dimension with a
            uniform velocity from the cradle to the grave. Just as we should travel _down_ if we
            began our existence fifty miles above the earth’s surface.”
        """,
    },
    {
        "file": "human_14.txt",
        "title": "The Time Machine",
        "author": "H. G. Wells",
        "year": 1895,
        "gutenberg_id": 35,
        "text": """
            “Really and truly I do.” And he looked frankly into my eyes. He hesitated. His eye
            wandered about the room. “I only want half an hour,” he said. “I know why you came, and
            it’s awfully good of you. There’s some magazines here. If you’ll stop to lunch I’ll prove
            you this time travelling up to the hilt, specimens and all. If you’ll forgive my leaving
            you now?”
        """,
    },
    {
        "file": "human_15.txt",
        "title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "year": 1925,
        "gutenberg_id": 64317,
        "text": """
            We walked through a high hallway into a bright rosy-coloured space, fragilely bound into
            the house by French windows at either end. The windows were ajar and gleaming white
            against the fresh grass outside that seemed to grow a little way into the house. A breeze
            blew through the room, blew curtains in at one end and out the other like pale flags,
            twisting them up toward the frosted wedding-cake of the ceiling, and then rippled over
            the wine-coloured rug, making a shadow on it as wind does on the sea.
        """,
    },
    {
        "file": "human_16.txt",
        "title": "The Great Gatsby",
        "author": "F. Scott Fitzgerald",
        "year": 1925,
        "gutenberg_id": 64317,
        "text": """
            It made no difference to me. Dishonesty in a woman is a thing you never blame deeply—I
            was casually sorry, and then I forgot. It was on that same house-party that we had a
            curious conversation about driving a car. It started because she passed so close to some
            workmen that our fender flicked a button on one man’s coat.
        """,
    },
    {
        "file": "human_17.txt",
        "title": "The Wonderful Wizard of Oz",
        "author": "L. Frank Baum",
        "year": 1900,
        "gutenberg_id": 55,
        "text": """
            From the far north they heard a low wail of the wind, and Uncle Henry and Dorothy could
            see where the long grass bowed in waves before the coming storm. There now came a sharp
            whistling in the air from the south, and as they turned their eyes that way they saw
            ripples in the grass coming from that direction also.
        """,
    },
    {
        "file": "human_18.txt",
        "title": "The Wonderful Wizard of Oz",
        "author": "L. Frank Baum",
        "year": 1900,
        "gutenberg_id": 55,
        "text": """
            “I am now worse off than when I first met Dorothy,” he thought. “Then, I was stuck on a
            pole in a cornfield, where I could make-believe scare the crows, at any rate. But surely
            there is no use for a Scarecrow stuck on a pole in the middle of a river. I am afraid I
            shall never have any brains, after all!”
        """,
    },
    {
        "file": "human_19.txt",
        "title": "The Call of the Wild",
        "author": "Jack London",
        "year": 1903,
        "gutenberg_id": 215,
        "text": """
            Four men gingerly carried the crate from the wagon into a small, high-walled back yard.
            A stout man, with a red sweater that sagged generously at the neck, came out and signed
            the book for the driver. That was the man, Buck divined, the next tormentor, and he
            hurled himself savagely against the bars. The man smiled grimly, and brought a hatchet
            and a club.
        """,
    },
    {
        "file": "human_20.txt",
        "title": "The Call of the Wild",
        "author": "Jack London",
        "year": 1903,
        "gutenberg_id": 215,
        "text": """
            Nobody spoke. Thornton’s bluff, if bluff it was, had been called. He could feel a flush
            of warm blood creeping up his face. His tongue had tricked him. He did not know whether
            Buck could start a thousand pounds. Half a ton! The enormousness of it appalled him. He
            had great faith in Buck’s strength and had often thought him capable of starting such a
            load; but never, as now, had he faced the possibility of it, the eyes of a dozen men
            fixed upon him, silent and waiting. Further, he had no thousand dollars; nor had Hans or
            Pete.
        """,
    },
]


# --- AI-style text: 20 agent-composed ChatGPT-style outputs -----------------
# Provenance (honest): written by the coding agent (GLM, this coding session)
# in a ChatGPT-typical assistant register for this task; NOT collected ChatGPT
# traffic and NOT verified against any real ChatGPT transcript.

_AI_SOURCE = (
    "agent-composed (coding agent, this session); ChatGPT-style register; not real ChatGPT traffic"
)

TEXT_AI: list[dict[str, Any]] = [
    {
        "file": "ai_01.txt",
        "text": """
            Certainly! Here's a simple weeknight garlic butter shrimp recipe:

            Ingredients:
            - 1 lb large shrimp, peeled and deveined
            - 4 tbsp butter
            - 4 cloves garlic, minced
            - Juice of half a lemon
            - Salt, pepper, and chopped parsley

            Instructions:
            1. Pat the shrimp dry and season with salt and pepper.
            2. Melt the butter over medium heat, then sauté the garlic for about 30 seconds.
            3. Add the shrimp and cook 2 minutes per side, until pink and opaque.
            4. Finish with lemon juice and parsley, toss, and serve immediately.

            This pairs beautifully with rice or crusty bread. Let me know if you'd like a
            dairy-free or spicier variation!
        """,
    },
    {
        "file": "ai_02.txt",
        "text": """
            Of course! Here's a solid packing list for a 5-day city trip:

            - 5 tops and 2 bottoms (mix and match)
            - 1 light jacket or blazer
            - Comfortable walking shoes
            - Toiletry kit (travel sizes)
            - Phone charger and a power bank
            - Reusable water bottle
            - Copies of important documents

            Pro tip: roll your clothes instead of folding — it saves space and reduces wrinkles.
            Would you like me to adjust this for warm or cold weather?
        """,
    },
    {
        "file": "ai_03.txt",
        "text": """
            Here's a polite follow-up email you can customize:

            Subject: Following up on our conversation

            Hi [Name],

            I hope your week is going well. I wanted to briefly follow up on our conversation
            from last Tuesday regarding the proposed timeline.

            Whenever you have a moment, I'd appreciate your thoughts — even a quick reply is
            helpful. If it's easier, I'm happy to set up a short call.

            Thank you again for your time, and I look forward to hearing from you.

            Best regards,
            [Your name]

            Feel free to tell me the context and I'll tailor it further.
        """,
    },
    {
        "file": "ai_04.txt",
        "text": """
            Great question! In simple terms, photosynthesis is how plants make their own food
            using sunlight. There are two main parts:

            1. Light-dependent reactions — the plant captures sunlight with chlorophyll and uses
               it to split water molecules, releasing oxygen as a byproduct.
            2. The Calvin cycle — the plant takes in carbon dioxide from the air and uses the
               energy captured earlier to build glucose, a sugar it stores for energy.

            The overall equation is: carbon dioxide + water + light → glucose + oxygen.

            In short: plants breathe in CO2, drink water, catch sunlight, and produce sugar and
            the oxygen we breathe. Let me know if you'd like a diagram-friendly version!
        """,
    },
    {
        "file": "ai_05.txt",
        "text": """
            Absolutely — here's a cover letter draft you can personalize:

            Dear Hiring Manager,

            I'm excited to apply for the Marketing Coordinator role at [Company]. With three
            years of experience running multi-channel campaigns and a track record of growing
            engagement by 40% at my current role, I believe I'd be a strong addition to your
            team.

            What draws me to [Company] is your focus on data-driven storytelling — an approach
            I've used to turn campaign metrics into decisions that moved revenue.

            I'd welcome the chance to discuss how my background aligns with your goals.

            Sincerely,
            [Your name]

            Want me to tune it to the specific job description? Just paste it here.
        """,
    },
    {
        "file": "ai_06.txt",
        "text": """
            Sure! Here's a balanced beginner-friendly weekly workout plan:

            Monday — Full-body strength: squats, push-ups, rows, plank (3 sets each)
            Tuesday — 30 minutes brisk walk or light jog
            Wednesday — Rest or gentle stretching
            Thursday — Strength: lunges, overhead press, glute bridges, dead bug
            Friday — 30 minutes cycling or swimming
            Saturday — Active fun: hike, sports, or a long walk
            Sunday — Rest

            Start lighter than feels necessary and add a little each week. If you have any
            health conditions, check with a professional first. Want me to adapt this for
            home workouts with no equipment?
        """,
    },
    {
        "file": "ai_07.txt",
        "text": """
            The error you're seeing usually means you're passing a `str` where the code expects
            `bytes` (or the reverse). Here's how to fix it:

            1. Check the failing line and print `type(value)` for each argument.
            2. If you open a file that must be text, use `open(path, "r", encoding="utf-8")`.
            3. If an API returns bytes, decode first: `data.decode("utf-8")`.
            4. When encoding, prefer explicit targets: `text.encode("utf-8")`.

            In most codebases this shows up at I/O boundaries — file reads, sockets, and HTTP
            responses. Add a tiny helper like `ensure_str(x)` and use it at those boundaries so
            the rest of the code can stay consistent.

            If you paste the full traceback, I can point at the exact line to change.
        """,
    },
    {
        "file": "ai_08.txt",
        "text": """
            In summary, the novel explores three interlocking themes:

            1. Memory and identity — the narrator's recollections are unreliable, so the reader
               must decide what actually happened versus what was invented to make loss bearable.
            2. Class and aspiration — the protagonist's rise is measured in objects and accents,
               and every gained status marker quietly costs a relationship.
            3. The unrecoverable past — the ending suggests that returning home is a mirage; the
               town has changed, and so has the person remembering it.

            The prose stays plain and restrained, which makes the final chapter's confession land
            harder. If you'd like, I can pull out key quotes for each theme or compare it with a
            similar book.
        """,
    },
    {
        "file": "ai_09.txt",
        "text": """
            Here are some thoughtful gift ideas for someone who loves hiking:

            - A trail map case or a national parks annual pass
            - Merino wool hiking socks (a universally appreciated upgrade)
            - A lightweight packable down jacket
            - A telescoping trekking pole pair
            - A compact water filter for day hikes
            - A book of local trail guides, or a subscription to a trail app

            If you share a budget and their shoe or jacket size, I can narrow this to a single
            best pick and a runner-up.
        """,
    },
    {
        "file": "ai_10.txt",
        "text": """
            When choosing a blender, weigh these four factors:

            1. Motor power — 700–1000 W is fine for smoothies; 1200 W+ handles ice and nut butters.
            2. Blade and jar design — angled stainless blades plus a tapered jar reduce pockets
               that stop the vortex.
            3. Noise — personal blenders are louder per watt; look for a sound shield if you blend
               early in the morning.
            4. Cleaning — dishwasher-safe, wide-base jars save the most time.

            For most households a mid-range 1000 W model covers 95% of tasks. Tell me what you
            blend most often and I'll recommend a specific tier.
        """,
    },
    {
        "file": "ai_11.txt",
        "text": """
            You can accomplish this with a single JOIN:

            SELECT o.id, o.total, c.name
            FROM orders o
            JOIN customers c ON c.id = o.customer_id
            WHERE o.created_at >= DATE '2026-01-01'
              AND o.status = 'paid'
            ORDER BY o.created_at DESC;

            Two notes: make sure there's an index on `orders(customer_id, created_at)` — the
            planner will use it to avoid a scan — and if a customer can have no orders yet but
            you still want them listed, switch to a LEFT JOIN from customers to orders.

            Happy to adapt this to your exact schema if you paste the table definitions.
        """,
    },
    {
        "file": "ai_12.txt",
        "text": """
            Sure — here's a 2-week study plan for the exam:

            Week 1: foundations
            - Days 1–3: re-read chapters 1–3 and write a one-page summary each
            - Days 4–5: work 20 practice problems on the core formulas
            - Day 6: review mistakes and redo the hardest three
            - Day 7: rest day (light flashcards only)

            Week 2: exam conditions
            - Days 8–10: two timed practice papers
            - Days 11–12: fix the two weakest topics only
            - Day 13: full mock exam at the real start time
            - Day 14: skim summaries, sleep early

            The pattern is: learn → practice → review → simulate. Want a version with daily
            time blocks?
        """,
    },
    {
        "file": "ai_13.txt",
        "text": """
            Here's a sincere, professional apology note:

            Subject: My apologies for yesterday

            Hi [Name],

            I want to apologize for missing yesterday's review session. There's no excuse — I
            misread the time, and that's on me.

            I know the meeting ran short without a full quorum, and I don't want my mistake to
            cost the team progress. I've already reviewed the notes and prepared my feedback;
            I'll share it in writing today so nobody is blocked.

            Again, I'm sorry. It won't happen again.

            Best,
            [Your name]

            I can make it warmer, shorter, or more formal — just say the word.
        """,
    },
    {
        "file": "ai_14.txt",
        "text": """
            Here's a haiku about the first snowfall, plus a quick explanation:

            First snow holds the street —
            every parked car wears a soft
            white hood, hushed and still.

            The poem follows the 5-7-5 syllable pattern and leans on a single image: cars
            "wearing" snow like hoods, which turns the ordinary street into something patient
            and quiet. The dash after the first line creates the pause that mirrors the hush.

            Want another version — playful instead of quiet, or one about spring rain?
        """,
    },
    {
        "file": "ai_15.txt",
        "text": """
            Happy to help — here are stronger versions of your bullet points:

            Before: "Responsible for managing the team's social media accounts."
            After: "Grew the team's social presence to 30k followers in 12 months by shipping a
            weekly content series and cutting response time from 24h to 3h."

            Before: "Helped with the migration project."
            After: "Led the zero-downtime migration of 1.2M records to the new billing system,
            coordinating 3 teams and finishing two sprints early."

            The formula: strong verb + what you did + measurable result. Send me the rest of
            your resume bullets and I'll rewrite them in the same shape.
        """,
    },
    {
        "file": "ai_16.txt",
        "text": """
            Of course — here's a short bedtime story:

            The Lantern Who Was Afraid of the Dark

            In a small harbor town lived a lantern named Pip. Every evening the other lamps
            glowed proudly, but Pip trembled, because inside him the flame was small.

            One night a fog rolled in, thicker than wool, and the big lamps flickered out. A
            lost kitten mewed at the end of the pier. Pip took a deep breath and let his little
            flame stand up as tall as it could — and it was just enough. Step by step, the
            kitten followed the light home.

            The harbor master said, "Pip, you were the brightest lamp tonight." Pip learned
            that courage isn't a big flame; it's a small one that refuses to go out.

            The end. Want a longer version or a different animal tomorrow night?
        """,
    },
    {
        "file": "ai_17.txt",
        "text": """
            Before you file, make sure you have these documents ready:

            - Last year's tax return (for carryovers and comparison)
            - All W-2s and 1099s (including interest and dividend forms)
            - Records of estimated tax payments you made during the year
            - Receipts for deductible expenses: charitable donations, medical costs above the
              threshold, and eligible education expenses
            - Your bank routing details for direct deposit or payment

            Also check: did you change jobs, move states, or get married? Any of those changes
            your withholding and credits. I'm not a tax professional and this isn't tax advice —
            for your specific situation, confirm with a licensed preparer.

            Would you like a printable checklist version of this list?
        """,
    },
    {
        "file": "ai_18.txt",
        "text": """
            I'm sorry to hear you're having trouble with your order — I completely understand how
            frustrating that is, and I'll make sure we get it sorted out quickly.

            Here's what I can see on your account: order #48213 shipped on the 3rd and is
            currently showing as in transit with the carrier, though the tracking hasn't
            updated in five days. That's longer than normal, so I've flagged it with the
            carrier and requested an expedited trace.

            In the meantime, here's what I've done for you: I've applied store credit for the
            shipping delay, and if the package isn't scanned within 48 hours, I'll send a free
            replacement immediately — no need to return anything.

            Would you like me to follow up by email as soon as the trace completes?
        """,
    },
    {
        "file": "ai_19.txt",
        "text": """
            Great question! The sky is blue because of the way sunlight bounces around in the
            air.

            Sunlight looks white, but it's really made of all colors mixed together. When it
            enters the atmosphere, it bumps into tiny gas molecules. Blue light travels in
            shorter, choppier waves, so it gets bounced around far more than other colors and
            scatters across the whole sky — that scattered blue light is what reaches your eyes
            from every direction.

            At sunset, light has to travel through much more air, so most of the blue gets
            scattered away before it reaches you, leaving the reds and oranges behind — which
            is why sunsets glow warm.

            Want a simple experiment you can do with a flashlight and a glass of milky water?
        """,
    },
    {
        "file": "ai_20.txt",
        "text": """
            Here's a relaxed 3-day Lisbon itinerary:

            Day 1 — Historic center: start in Baixa, ride Tram 28 to Alfama, get lost in its
            alleys, then visit São Jorge Castle before sunset. Dinner: a tascas-style spot with
            grilled sardines.

            Day 2 — Belém: pastéis de nata at the original bakery, then the Jerónimos Monastery,
            the Tower of Belém, and the riverside walk. Afternoon: the MAAT museum.

            Day 3 — Views and neighborhoods: Miradouro da Senhora do Monte in the morning, lunch
            in Príncipe Real, then the LX Factory for shops and street art. Finish with fado in
            Bairro Alto.

            Pace is deliberately light — two anchors per day, not five. Want me to swap a day
            for a Sintra side trip?
        """,
    },
]


# --- image sources -----------------------------------------------------------

NASA_BASE = "https://images-assets.nasa.gov/image"
NASA_PAGE = "https://images.nasa.gov/details"
SM_SPACE = "https://huggingface.co/spaces/abacusai/Smaug-AI-Image-Detector/resolve/main/examples"

# Provenance keys: file, url, page (human-readable provenance page), origin,
# license, note. "Tool" identifies the AI generator where applicable.
IMAGE_SOURCES: list[dict[str, str]] = [
    # --- real (NASA image library, public domain; NASA imagery is generally not
    # protected by copyright — see images.nasa.gov usage guidelines) ---
    {
        "file": "real_01.jpg",
        "url": f"{NASA_BASE}/GSFC_20171208_Archive_e001386/GSFC_20171208_Archive_e001386~large.jpg",
        "page": f"{NASA_PAGE}/GSFC_20171208_Archive_e001386",
        "origin": "NASA images library",
        "license": "public domain (NASA media usage guidelines)",
        "note": "Blue Marble 2012, Earth from space",
    },
    {
        "file": "real_02.jpg",
        "url": f"{NASA_BASE}/iss072e487421/iss072e487421~large.jpg",
        "page": f"{NASA_PAGE}/iss072e487421",
        "origin": "NASA images library",
        "license": "public domain (NASA media usage guidelines)",
        "note": "Astronaut spacewalk outside the ISS",
    },
    {
        "file": "real_03.jpg",
        "url": f"{NASA_BASE}/iss072e159172/iss072e159172~large.jpg",
        "page": f"{NASA_PAGE}/iss072e159172",
        "origin": "NASA images library",
        "license": "public domain (NASA media usage guidelines)",
        "note": "Aurora borealis over Earth from the ISS",
    },
    {
        "file": "real_04.jpg",
        "url": f"{NASA_BASE}/PIA24542/PIA24542~large.jpg",
        "page": f"{NASA_PAGE}/PIA24542",
        "origin": "NASA images library",
        "license": "public domain (NASA media usage guidelines)",
        "note": "Perseverance rover selfie with Ingenuity helicopter (Mars)",
    },
    {
        "file": "real_05.jpg",
        "url": f"{NASA_BASE}/PIA07081/PIA07081~orig.jpg",
        "page": f"{NASA_PAGE}/PIA07081",
        "origin": "NASA images library",
        "license": "public domain (NASA media usage guidelines)",
        "note": "Mars Exploration Rover studying Martian soil",
    },
    {
        "file": "real_06.jpg",
        "url": f"{NASA_BASE}/iss072e001649/iss072e001649~large.jpg",
        "page": f"{NASA_PAGE}/iss072e001649",
        "origin": "NASA images library",
        "license": "public domain (NASA media usage guidelines)",
        "note": "Hurricane Helene photographed from the ISS",
    },
    {
        "file": "real_07.jpg",
        "url": f"{NASA_BASE}/AFRC2025-0168-028/AFRC2025-0168-028~large.jpg",
        "page": f"{NASA_PAGE}/AFRC2025-0168-028",
        "origin": "NASA images library",
        "license": "public domain (NASA media usage guidelines)",
        "note": "NASA X-59 supersonic research aircraft in flight",
    },
    {
        "file": "real_08.jpg",
        "url": f"{NASA_BASE}/GSFC_20171208_Archive_e001210/GSFC_20171208_Archive_e001210~large.jpg",
        "page": f"{NASA_PAGE}/GSFC_20171208_Archive_e001210",
        "origin": "NASA images library",
        "license": "public domain (NASA media usage guidelines)",
        "note": "Solar flare observed by SDO",
    },
    {
        "file": "real_09.jpg",
        "url": f"{NASA_BASE}/GSFC_20171208_Archive_e000603/GSFC_20171208_Archive_e000603~orig.jpg",
        "page": f"{NASA_PAGE}/GSFC_20171208_Archive_e000603",
        "origin": "NASA images library",
        "license": "public domain (NASA media usage guidelines)",
        "note": "Super Blood Moon lunar eclipse",
    },
    {
        "file": "real_10.jpg",
        "url": f"{SM_SPACE}/real_photo.jpg",
        "page": "https://huggingface.co/spaces/abacusai/Smaug-AI-Image-Detector",
        "origin": "HF Space abacusai/Smaug-AI-Image-Detector examples/ (model author demo)",
        "license": "demo fixture shipped by the model author (Space repo)",
        "note": "clearly-photographic demo image (same file as the T10 smoke)",
    },
    # --- AI-generated (Wikimedia Commons per-tool categories + one Space demo) ---
    # Commons URLs use the standard 1280px thumbnail endpoints (the CDN-cached
    # pattern Wikimedia's 429 advice recommends) so the fetch is small and
    # polite; provenance pages below are the original file pages.
    {
        "file": "ai_01.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e8/"
        "%27Greenwood_Estates_Vista_City%27_by_Midjourney.jpg/"
        "1280px-%27Greenwood_Estates_Vista_City%27_by_Midjourney.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:'Greenwood_Estates_Vista_City'_by_Midjourney.jpg",
        "origin": "Wikimedia Commons, Category:Images generated by Midjourney",
        "license": "CC0 (per Commons file page)",
        "note": "Midjourney",
    },
    {
        "file": "ai_02.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/43/"
        "%27Night_898%27_fantasy_past_city_by_Midjourney.jpg/"
        "1280px-%27Night_898%27_fantasy_past_city_by_Midjourney.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:'Night_898'_fantasy_past_city_by_Midjourney.jpg",
        "origin": "Wikimedia Commons, Category:Images generated by Midjourney",
        "license": "CC0 (per Commons file page)",
        "note": "Midjourney",
    },
    {
        "file": "ai_03.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ed/"
        "Scenic_Forest_Valley_in_the_Late_Afternoon_%28FLUX.1%29.jpg/"
        "1280px-Scenic_Forest_Valley_in_the_Late_Afternoon_%28FLUX.1%29.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:Scenic_Forest_Valley_in_the_Late_Afternoon_(FLUX.1).jpg",
        "origin": "Wikimedia Commons, Category:Images generated by Flux (text-to-image model)",
        "license": "public domain (per Commons file page)",
        "note": "FLUX.1",
    },
    {
        "file": "ai_04.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/"
        "Sunset_Valley_%28FLUX_1.1_Pro_Raw%29.webp/"
        "1280px-Sunset_Valley_%28FLUX_1.1_Pro_Raw%29.webp",
        "page": "https://commons.wikimedia.org/wiki/File:Sunset_Valley_(FLUX_1.1_Pro_Raw).webp",
        "origin": "Wikimedia Commons, Category:Images generated by Flux (text-to-image model)",
        "license": "public domain (per Commons file page)",
        "note": "FLUX 1.1 Pro Raw (webp, converted to RGB JPEG here)",
    },
    {
        "file": "ai_05.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/04/"
        "A_professional_photoshoot_show_a_anthropomorphic_rhinoceros_wearing_business_suit_"
        "standing_in_front_of_the_cinema_near_the_Broadway%2C_New_York_City._The_cinema_had_"
        "a_marquee_at_the_entrance_with_the_text_%22NOW_SHOWING_FLUX_1.0%22.png/"
        "1280px-A_professional_photoshoot_show_a_anthropomorphic_rhinoceros_wearing_business_"
        "suit_standing_in_front_of_the_cinema_near_the_Broadway%2C_New_York_City._The_cinema_"
        "had_a_marquee_at_the_entrance_with_the_text_%22NOW_SHOWING_FLUX_1.0%22.png",
        "page": "https://commons.wikimedia.org/wiki/File:A_professional_photoshoot_show_a_"
        "anthropomorphic_rhinoceros_wearing_business_suit_standing_in_front_of_the_cinema_near_"
        "the_Broadway,_New_York_City._The_cinema_had_a_marquee_at_the_entrance_with_the_text_"
        "%22NOW_SHOWING_FLUX_1.0%22.png",
        "origin": "Wikimedia Commons, Category:Images generated by Flux (text-to-image model)",
        "license": "public domain (per Commons file page)",
        "note": "FLUX",
    },
    {
        "file": "ai_06.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5f/"
        "A_golden_skin_lady_wearing_ornamental_heads_and_golden_ornaments_on_her_body_standing_"
        "in_a_lake.jpg/1280px-A_golden_skin_lady_wearing_ornamental_heads_and_golden_ornaments_"
        "on_her_body_standing_in_a_lake.jpg",
        "page": "https://commons.wikimedia.org/wiki/"
        "File:A_golden_skin_lady_wearing_ornamental_heads_and_golden_ornaments_on_her_body_"
        "standing_in_a_lake.jpg",
        "origin": "Wikimedia Commons, Category:Images generated by DALL-E",
        "license": "CC BY-SA 4.0 (per Commons file page)",
        "note": "DALL-E",
    },
    {
        "file": "ai_07.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/"
        "AI_generated_image_of_a_cancer_cell.jpg/"
        "1280px-AI_generated_image_of_a_cancer_cell.jpg",
        "page": "https://commons.wikimedia.org/wiki/File:AI_generated_image_of_a_cancer_cell.jpg",
        "origin": "Wikimedia Commons, Category:Images generated by DALL-E",
        "license": "CC BY-SA 4.0 (per Commons file page)",
        "note": "DALL-E (scientific-illustration style)",
    },
    {
        "file": "ai_08.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/"
        "A_colonized_comet_with_a_long_abandoned_reception_center_above_underground_facility_"
        "%28AI_art%29.jpg/1280px-A_colonized_comet_with_a_long_abandoned_reception_center_above_"
        "underground_facility_%28AI_art%29.jpg",
        "page": "https://commons.wikimedia.org/wiki/"
        "File:A_colonized_comet_with_a_long_abandoned_reception_center_above_underground_"
        "facility_(AI_art).jpg",
        "origin": "Wikimedia Commons, Category:Images generated by Stable Diffusion",
        "license": "CC0 (per Commons file page)",
        "note": "Stable Diffusion",
    },
    {
        "file": "ai_09.jpg",
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/"
        "A_pod_plane_%28flying_car%29_recharging_at_a_wind_turbine_with_low_costs_during_excess_"
        "generation.jpg/1280px-A_pod_plane_%28flying_car%29_recharging_at_a_wind_turbine_with_"
        "low_costs_during_excess_generation.jpg",
        "page": "https://commons.wikimedia.org/wiki/"
        "File:A_pod_plane_(flying_car)_recharging_at_a_wind_turbine_with_low_costs_during_"
        "excess_generation.jpg",
        "origin": "Wikimedia Commons, Category:Images generated by Stable Diffusion",
        "license": "CC0 (per Commons file page)",
        "note": "Stable Diffusion",
    },
    {
        "file": "ai_10.jpg",
        "url": f"{SM_SPACE}/ai_image.jpg",
        "page": "https://huggingface.co/spaces/abacusai/Smaug-AI-Image-Detector",
        "origin": "HF Space abacusai/Smaug-AI-Image-Detector examples/ (model author demo)",
        "license": "demo fixture shipped by the model author (Space repo)",
        "note": "AI-generated demo image (same file as the T10 smoke)",
    },
]


# --- generation ---------------------------------------------------------------


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_text_fixtures(dest: Path) -> list[Path]:
    """Write the 40 text fixture files; return their paths."""
    paths: list[Path] = []
    for subdir, entries in (("human", TEXT_HUMAN), ("ai", TEXT_AI)):
        directory = dest / "text" / subdir
        directory.mkdir(parents=True, exist_ok=True)
        for entry in entries:
            path = directory / entry["file"]
            path.write_text(_flatten(entry["text"]).strip() + "\n", encoding="utf-8")
            paths.append(path)
    return paths


def fetch_image(url: str, attempts: int = 6) -> Image.Image:
    """Download ``url`` and return the decoded PIL image (429/5xx backoff retry).

    Wikimedia Commons rate-limits rapid successive downloads (HTTP 429), so
    failures back off 20 s * attempt and retry; other errors raise immediately.
    """
    for attempt in range(1, attempts + 1):
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 500, 502, 503) and attempt < attempts:
                time.sleep(20 * attempt)
                continue
            raise
        return Image.open(io.BytesIO(payload))
    raise RuntimeError(f"unreachable: retries exhausted for {url}")


def process_image(image: Image.Image, max_side: int, quality: int) -> Image.Image:
    """EXIF-transpose, composite onto white, convert to RGB, cap the long side."""
    prepared = ImageOps.exif_transpose(image)
    if prepared.mode not in ("RGB", "L"):
        background = Image.new("RGB", prepared.size, (255, 255, 255))
        rgba = prepared.convert("RGBA")
        background.paste(rgba, mask=rgba.split()[-1])
        prepared = background
    else:
        prepared = prepared.convert("RGB")
    if min(prepared.size) < SMALL_PX:
        raise ValueError(f"source too small: {prepared.size} (short side < {SMALL_PX}px)")
    prepared.thumbnail((max_side, max_side), Image.LANCZOS)
    return prepared


def write_image_fixtures(dest: Path, max_side: int, quality: int, force: bool) -> list[Path]:
    """Download, process and write the 20 image fixture files; return their paths.

    Existing files are skipped unless ``force`` is set (resumable runs: a
    rate-limited Wikimedia fetch must not re-download the whole set).
    """
    paths: list[Path] = []
    for subdir, wanted in (("real", "real_"), ("ai", "ai_")):
        directory = dest / "images" / subdir
        directory.mkdir(parents=True, exist_ok=True)
        for source in IMAGE_SOURCES:
            if not source["file"].startswith(wanted):
                continue
            path = directory / source["file"]
            if path.exists() and not force:
                paths.append(path)
                print(f"  {source['file']}: kept (exists; use --force to re-fetch)")
                continue
            processed = process_image(fetch_image(source["url"]), max_side, quality)
            processed.save(path, "JPEG", quality=quality, optimize=True)
            paths.append(path)
            print(f"  {source['file']}: {processed.size[0]}x{processed.size[1]} <- {source['url']}")
            time.sleep(4.0)  # pacing: Commons rate-limits rapid successive hits
    return paths


def write_manifest(paths: list[Path], dest: Path, max_side: int, quality: int) -> Path:
    """Write MANIFEST.md with provenance + sha256 for every generated file."""
    by_name = {path.name: path for path in paths}
    lines = [
        "# Eval fixture provenance (`data/eval`) — Task 11 bake-off",
        "",
        (
            f"Generated {datetime.now(UTC).isoformat()} by "
            f"`backend/tests/fixtures/make_fixtures.py` (images: long side {max_side}px, "
            f"JPEG q{quality})."
        ),
        "",
        "Verdict bands used by the bake-off (app settings): authentic < 35,",
        "unclear 35-65, AI >= 65 (settings.band_unclear_min / band_synthetic_min).",
        "",
        "## Text — human (`text/human/`, 20 files)",
        "",
        "20 single-paragraph excerpts (280-620 chars, whitespace-collapsed to",
        "single spaces) from public-domain Project Gutenberg plain-text editions",
        "(all pre-1930, therefore pre-AI). Provenance = book + author + year +",
        "ebook id; source URL pattern: `https://www.gutenberg.org/ebooks/<id>`",
        "(plain text: `https://www.gutenberg.org/cache/epub/<id>/pg<id>.txt`).",
        "",
        "| file | title | author | year | gutenberg id | sha256 |",
        "|---|---|---|---|---|---|",
    ]
    for entry in TEXT_HUMAN:
        path = by_name[entry["file"]]
        lines.append(
            f"| {entry['file']} | {entry['title']} | {entry['author']} | {entry['year']} "
            f"| {entry['gutenberg_id']} | `{sha256_of(path)}` |"
        )
    lines += [
        "",
        "## Text — AI-style (`text/ai/`, 20 files)",
        "",
        f"All 20 files: {_AI_SOURCE}. Composed by the coding agent for this",
        "bake-off in a ChatGPT-typical assistant register (polite opener,",
        "structure, closing offer). NOT collected ChatGPT traffic.",
        "",
        "| file | sha256 |",
        "|---|---|",
    ]
    for entry in TEXT_AI:
        path = by_name[entry["file"]]
        lines.append(f"| {entry['file']} | `{sha256_of(path)}` |")
    lines += [
        "",
        "## Images — real (`images/real/`, 10 files)",
        "",
        "9 NASA image-library photographs (public domain) + 1 model-author demo",
        "photograph. All are real photographs of real subjects.",
        "",
        "| file | origin | note | license | source page | sha256 |",
        "|---|---|---|---|---|---|",
    ]
    for source in IMAGE_SOURCES:
        if not source["file"].startswith("real_"):
            continue
        path = by_name[source["file"]]
        lines.append(
            f"| {source['file']} | {source['origin']} | {source['note']} | {source['license']} "
            f"| {source['page']} | `{sha256_of(path)}` |"
        )
    lines += [
        "",
        "## Images — AI-generated (`images/ai/`, 10 files)",
        "",
        "9 Wikimedia Commons AI-generated images (per-tool categories, license",
        "as stated on each Commons file page) + 1 model-author demo AI image.",
        "",
        "| file | origin | note | license | source page | sha256 |",
        "|---|---|---|---|---|---|",
    ]
    for source in IMAGE_SOURCES:
        if not source["file"].startswith("ai_"):
            continue
        path = by_name[source["file"]]
        lines.append(
            f"| {source['file']} | {source['origin']} | {source['note']} | {source['license']} "
            f"| {source['page']} | `{sha256_of(path)}` |"
        )
    lines += [
        "",
        "## Regeneration caveats",
        "",
        "- Text fixtures are embedded in the generator (offline, byte-stable).",
        "- Image fixtures are downloaded; the processing above is deterministic",
        "  so the same source bytes reproduce the same sha256. Commons sources",
        "  use the CDN-cached 1280px thumbnail endpoints. If a source URL rots,",
        "  replace the entry in IMAGE_SOURCES (and note it in the file's",
        "  provenance) — provenance travels with the manifest, not the URL.",
        "- The generator is not collected by pytest; the offline suite never",
        "  touches the network.",
        "",
    ]
    manifest = dest / "MANIFEST.md"
    manifest.write_text("\n".join(lines), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text-only", action="store_true", help="skip image downloads")
    parser.add_argument("--images-only", action="store_true", help="skip text fixtures")
    parser.add_argument("--force", action="store_true", help="re-fetch images even if files exist")
    parser.add_argument("--max-side", type=int, default=1280, help="long side cap (default 1280)")
    parser.add_argument("--quality", type=int, default=88, help="JPEG quality (default 88)")
    args = parser.parse_args()
    if args.text_only and args.images_only:
        parser.error("--text-only and --images-only are mutually exclusive")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []
    if not args.images_only:
        print("writing text fixtures ...")
        generated += write_text_fixtures(EVAL_DIR)
    if not args.text_only:
        print("fetching + writing image fixtures ...")
        generated += write_image_fixtures(EVAL_DIR, args.max_side, args.quality, args.force)
    manifest = write_manifest(generated, EVAL_DIR, args.max_side, args.quality)
    print(f"wrote {len(generated)} fixture files + {manifest.name} into {EVAL_DIR}")
    if len(generated) != 60:
        sys.exit(f"expected 60 fixture files (40 text + 20 images), got {len(generated)}")


if __name__ == "__main__":
    main()
