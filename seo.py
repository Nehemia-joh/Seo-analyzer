import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from collections import Counter
import re
import time
import math
from lxml import etree
import xml.etree.ElementTree as ET
from textstat import flesch_kincaid_grade
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import urllib3
import logging
import langdetect
import csv
import json
import numpy as np
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from functools import lru_cache

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Set up logging
logging.basicConfig(filename='seo_analyzer.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def check_robots_txt(url):
    """Check if scraping is allowed by robots.txt"""
    try:
        rp = RobotFileParser()
        robots_url = urljoin(url, "/robots.txt")
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch("*", url)
    except Exception as e:
        logging.error(f"Error checking robots.txt for {url}: {e}")
        print(f"Error checking robots.txt: {e}")
        return True

@lru_cache(maxsize=100)
def get_page_content(url, retries=3, follow_redirects=True, use_head=False):
    """Fetch page content with retry logic and TTFB estimation"""
    headers = {'User-Agent': 'SEO-Analyzer-Bot/1.0 (Educational purpose)'}
    method = requests.head if use_head else requests.get
    ttfb = None
    for attempt in range(retries):
        try:
            start_time = time.time()
            response = method(url, headers=headers, timeout=10, verify=False, allow_redirects=follow_redirects)
            ttfb = time.time() - start_time
            response.raise_for_status()
            load_time = time.time() - start_time
            content = response.text if not use_head else ''
            return content, load_time, response.url.startswith('https://'), response.history, ttfb
        except requests.RequestException as e:
            logging.warning(f"Attempt {attempt+1} failed for {url}: {e}")
            if attempt == retries - 1:
                logging.error(f"Failed to fetch {url} after {retries} attempts: {e}")
                print(f"Error fetching {url}: {e}")
                return None, None, False, [], None
            time.sleep(2 ** attempt)
    return None, None, False, [], None

def check_broken_links(links, retries=3):
    """Check for broken links"""
    headers = {'User-Agent': 'SEO-Analyzer-Bot/1.0'}
    broken_links = []
    for link in links[:10]:
        for attempt in range(retries):
            try:
                response = requests.head(link, headers=headers, timeout=5, allow_redirects=True)
                if response.status_code >= 400:
                    broken_links.append((link, response.status_code))
                break
            except requests.RequestException:
                if attempt == retries - 1:
                    broken_links.append((link, 'Connection Error'))
                time.sleep(2 ** attempt)
        time.sleep(0.5)
    return broken_links

def get_sitemap_urls(url):
    """Check for sitemap and extract URLs"""
    sitemap_urls = []
    robots_url = urljoin(url, "/robots.txt")
    try:
        response = requests.get(robots_url, timeout=5, verify=False)
        if response.status_code == 200:
            for line in response.text.splitlines():
                if line.lower().startswith("sitemap:"):
                    sitemap_urls.append(line.split(":", 1)[1].strip())
        sitemap_url = urljoin(url, "/sitemap.xml")
        response = requests.get(sitemap_url, timeout=5, verify=False)
        if response.status_code == 200:
            sitemap_urls.append(sitemap_url)
            parser = etree.XMLParser(recover=True)
            root = etree.fromstring(response.content, parser=parser)
            sitemap_urls.extend([loc.text for loc in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}loc")])
        return list(set(sitemap_urls))[:10]
    except Exception as e:
        logging.warning(f"Failed to fetch sitemap for {url}: {e}")
        print(f"No sitemap found or server error: {e}")
        return sitemap_urls

def generate_sitemap(urls):
    """Generate a basic XML sitemap"""
    root = ET.Element("urlset", xmlns="http://www.sitemaps.org/schemas/sitemap/0.9")
    for url in urls:
        url_element = ET.SubElement(root, "url")
        loc = ET.SubElement(url_element, "loc")
        loc.text = url
        lastmod = ET.SubElement(url_element, "lastmod")
        lastmod.text = time.strftime("%Y-%m-%d")
        priority = ET.SubElement(url_element, "priority")
        priority.text = "0.8"
    xml_str = ET.tostring(root, encoding="UTF-8", xml_declaration=True).decode('utf-8')
    lines = xml_str.splitlines()
    indented = ['<?xml version="1.0" encoding="UTF-8"?>']
    indent_level = 0
    for line in lines[1:]:
        if line.startswith('</'):
            indent_level -= 1
        indented.append('  ' * indent_level + line)
        if line.startswith('<') and not line.startswith('</') and not line.endswith('/>'):
            indent_level += 1
    return '\n'.join(indented)

def save_sitemap(urls):
    """Save sitemap to a file"""
    try:
        sitemap_content = generate_sitemap(urls)
        with open('sitemap.xml', 'w', encoding='utf-8') as f:
            f.write(sitemap_content)
        print("Sitemap saved to 'sitemap.xml'.")
        logging.info("Sitemap generated and saved.")
    except Exception as e:
        logging.error(f"Error saving sitemap: {e}")
        print(f"Error saving sitemap: {e}")

def submit_sitemap_simulated(sitemap_url, site_url):
    """Simulate sitemap submission"""
    try:
        ping_url = f"https://www.google.com/ping?sitemap={sitemap_url}"
        response = requests.get(ping_url, timeout=5)
        if response.status_code == 200:
            print(f"Sitemap ping sent to Google for {sitemap_url}.")
            logging.info(f"Sitemap ping sent for {sitemap_url}.")
        else:
            print(f"Failed to ping Google for sitemap: HTTP {response.status_code}")
            logging.warning(f"Sitemap ping failed: HTTP {response.status_code}")
        print(f"To submit manually, go to Google Search Console (https://search.google.com/search-console), select '{site_url}', and submit '{sitemap_url}' under 'Sitemaps'.")
        logging.info(f"Manual sitemap submission instructions provided for {sitemap_url}.")
        return True
    except Exception as e:
        logging.error(f"Error pinging sitemap for {sitemap_url}: {e}")
        print(f"Error pinging sitemap: {e}")
        print(f"Manually submit sitemap at https://search.google.com/search-console.")
        return False

def check_indexing_status_simulated(url):
    """Estimate indexing status (limited to avoid Google limits)"""
    try:
        headers = {'User-Agent': 'SEO-Analyzer-Bot/1.0'}
        search_url = f"https://www.google.com/search?q=site:{url}"
        response = requests.get(search_url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('cite')
        is_indexed = any(url in cite.text for cite in results)
        status = "Likely Indexed" if is_indexed else "Not Found in Search Results"
        print(f"Indexing Status for {url}: {status}")
        logging.info(f"Simulated indexing status for {url}: {status}")
        return status
    except Exception as e:
        logging.error(f"Error checking indexing status for {url}: {e}")
        print(f"Error checking indexing status: {e}")
        return "Unknown"

def analyze_backlinks_simulated(url, pages):
    """Simulate backlink analysis (limited)"""
    try:
        domain = urlparse(url).netloc
        headers = {'User-Agent': 'SEO-Analyzer-Bot/1.0'}
        search_url = f"https://www.google.com/search?q=link:{domain}"
        response = requests.get(search_url, headers=headers, timeout=5)
        soup = BeautifulSoup(response.text, 'html.parser')
        results = soup.find_all('cite')
        referring_domains = set()
        for cite in results:
            link_domain = urlparse(cite.text).netloc
            if link_domain and link_domain != domain:
                referring_domains.add(link_domain)
        referring_domains = list(referring_domains)[:5]
        total_links = sum(len(page['external_links']) for page in pages)
        word_count_avg = sum(page['word_count'] for page in pages) / max(1, len(pages))
        domain_authority = min(100, len(referring_domains) * 5 + total_links // 2 + int(word_count_avg // 100))
        print(f"Referring Domains: {len(referring_domains)}")
        print(f"Sample Referring Domains: {referring_domains}")
        print(f"Estimated Domain Authority: {domain_authority}")
        logging.info(f"Backlink analysis for {url}: {len(referring_domains)} domains, DA {domain_authority}")
        return referring_domains, domain_authority
    except Exception as e:
        logging.error(f"Error analyzing backlinks for {url}: {e}")
        print(f"Error analyzing backlinks: {e}")
        return set(), 0

def post_to_x_simulated(summary):
    """Simulate X posting"""
    try:
        with open('seo_summary.txt', 'w', encoding='utf-8') as f:
            f.write(summary[:280])
        print("SEO summary saved to 'seo_summary.txt'. Copy and paste to post on X manually.")
        logging.info("SEO summary saved to seo_summary.txt")
        return True
    except Exception as e:
        logging.error(f"Error saving SEO summary: {e}")
        print(f"Error saving SEO summary: {e}")
        return False

def extract_links(soup, base_url):
    """Extract internal and external links with anchor text"""
    internal_links = []
    external_links = []
    anchor_texts = []
    domain = urlparse(base_url).netloc
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(base_url, href)
        link_domain = urlparse(full_url).netloc
        anchor = a_tag.get_text(strip=True)
        if link_domain == domain:
            internal_links.append(full_url)
            anchor_texts.append((full_url, anchor))
        elif link_domain:
            external_links.append(full_url)
    return list(set(internal_links)), list(set(external_links)), anchor_texts

def check_mobile_friendliness(soup):
    """Check for mobile-friendliness"""
    viewport = soup.find('meta', attrs={'name': re.compile(r'viewport', re.I)})
    if viewport and 'content' in viewport.attrs:
        content = viewport['content'].lower()
        if 'width=device-width' in content or 'initial-scale' in content:
            return True
    return False

def check_social_media_tags(soup):
    """Check for Open Graph and Twitter Card meta tags"""
    og_tags = soup.find_all('meta', attrs={'property': re.compile(r'^og:')})
    twitter_tags = soup.find_all('meta', attrs={'name': re.compile(r'^twitter:')})
    return len(og_tags) > 0, len(twitter_tags) > 0

def check_canonical_tag(soup, url):
    """Check for canonical tag"""
    canonical = soup.find('link', attrs={'rel': re.compile(r'canonical', re.I)})
    canonical_url = canonical['href'] if canonical and canonical.get('href') else None
    return canonical_url == url or urljoin(url, canonical_url) == url if canonical_url else False

def check_schema_markup(soup):
    """Check and validate schema.org structured data"""
    scripts = soup.find_all('script', type='application/ld+json')
    issues = []
    schema_types = []
    for script in scripts:
        try:
            data = json.loads(script.get_text(strip=True))
            schema_type = data.get('@type', 'Unknown')
            schema_types.append(schema_type)
            if schema_type not in ['Article', 'Product', 'Organization', 'WebPage']:
                issues.append(f"Uncommon schema type: {schema_type}")
        except json.JSONDecodeError as e:
            issues.append(f"Invalid JSON-LD: {str(e)}")
    return len(scripts) > 0, issues, schema_types

def check_robots_meta(soup):
    """Check for robots meta tags"""
    robots_tag = soup.find('meta', attrs={'name': re.compile(r'robots', re.I)})
    if robots_tag and robots_tag.get('content'):
        content = robots_tag['content'].lower()
        return {'noindex': 'noindex' in content, 'nofollow': 'nofollow' in content}
    return {'noindex': False, 'nofollow': False}

def check_favicon(soup, base_url):
    """Check for favicon"""
    favicon = soup.find('link', attrs={'rel': re.compile(r'(icon|shortcut icon)', re.I)})
    if favicon and favicon.get('href'):
        favicon_url = urljoin(base_url, favicon['href'])
        try:
            response = requests.head(favicon_url, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    return False

def check_meta_keywords(soup):
    """Check for meta keywords tag"""
    meta_keywords = soup.find('meta', attrs={'name': re.compile(r'keywords', re.I)})
    if meta_keywords and meta_keywords.get('content'):
        return [kw.strip() for kw in meta_keywords['content'].split(',')]
    return []

def check_image_optimization(soup, base_url):
    """Check image sizes and formats"""
    issues = []
    for img in soup.find_all('img')[:5]:
        src = img.get('src')
        if not src:
            continue
        img_url = urljoin(base_url, src)
        try:
            response = requests.head(img_url, timeout=5)
            if 'content-length' in response.headers:
                size_kb = int(response.headers['content-length']) / 1024
                if size_kb > 100:
                    issues.append(f"Image {img_url} is {size_kb:.1f} KB; consider compressing.")
            content_type = response.headers.get('content-type', '')
            if 'image/webp' not in content_type and 'image/avif' not in content_type:
                issues.append(f"Image {img_url} is not in WebP/AVIF; consider modern formats.")
        except requests.RequestException:
            issues.append(f"Image {img_url} could not be checked.")
    return issues

def check_breadcrumbs(soup):
    """Check for breadcrumb navigation"""
    return soup.find('script', type='application/ld+json', string=re.compile(r'BreadcrumbList')) or \
           soup.find(['nav', 'ol', 'ul'], attrs={'class': re.compile(r'breadcrumb', re.I)}) is not None

def check_hreflang_tags(soup):
    """Check for hreflang tags"""
    hreflang_tags = soup.find_all('link', attrs={'rel': 'alternate', 'hreflang': True})
    return len(hreflang_tags) > 0, [tag.get('hreflang') for tag in hreflang_tags]

def check_js_dependency(soup):
    """Check if critical content relies on JavaScript"""
    noscript_tags = soup.find_all('noscript')
    if noscript_tags:
        noscript_content = ' '.join(tag.get_text(strip=True) for tag in noscript_tags)
        return len(noscript_content) > 100
    return False

def accessibility_audit(soup):
    """Comprehensive WCAG 2.1 accessibility audit"""
    issues = set()
    images = soup.find_all('img')
    for img in images:
        if not img.get('alt'):
            issues.add(f"Image at {img.get('src', 'unknown')} lacks alt text (WCAG 1.1.1).")
        elif img.get('alt') == '':
            issues.add(f"Image at {img.get('src', 'unknown')} has empty alt text; use descriptive text or role='presentation' (WCAG 1.1.1).")
    videos = soup.find_all('video')
    for video in videos:
        if not video.find('track', kind='captions'):
            issues.add("Video lacks captions (WCAG 1.2.2).")
    headings = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']
    heading_levels = []
    for tag_name in headings:
        for _ in soup.find_all(tag_name):
            level = int(tag_name[1])
            heading_levels.append(level)
    if heading_levels:
        if heading_levels[0] != 1:
            issues.add("First heading is not H1 (WCAG 1.3.1).")
        for i in range(1, len(heading_levels)):
            if heading_levels[i] > heading_levels[i-1] + 1:
                issues.add(f"Heading level jump from H{heading_levels[i-1]} to H{heading_levels[i]} (WCAG 1.3.1).")
    else:
        issues.add("No headings found (WCAG 1.3.1).")
    text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'a'])
    for elem in text_elements[:10]:
        style = elem.get('style', '')
        if 'color' in style and 'background' not in style:
            issues.add(f"Element {elem.name} may lack sufficient contrast; specify background color (WCAG 1.4.3).")
    focusable = soup.find_all(['a', 'button', 'input', 'select', 'textarea'])
    for elem in focusable:
        if elem.get('tabindex') == '-1' and not elem.get('aria-hidden'):
            issues.add(f"Element {elem.name} is not keyboard accessible (WCAG 2.1.1).")
    if not soup.find('a', href='#main') and not soup.find('main', attrs={'role': 'main'}):
        issues.add("No skip link or main landmark found (WCAG 2.4.1).")
    if not soup.get('lang'):
        issues.add("HTML element lacks 'lang' attribute (WCAG 3.1.1).")
    aria_elements = soup.find_all(attrs={'aria-label': True, 'aria-labelledby': True, 'role': True})
    for elem in aria_elements:
        if not elem.get('role'):
            issues.add(f"Element with ARIA attributes lacks role (WCAG 4.1.2).")
    return list(issues)

def check_amp(soup, url):
    """Check for AMP compliance"""
    is_amp = soup.find('html', {'amp': True}) is not None or '⚡' in soup.attrs.get('amp', '')
    if not is_amp:
        return False, ["Not an AMP page; missing <html amp> or ⚡ attribute."]
    issues = []
    if not soup.find('meta', charset='utf-8'):
        issues.append("Missing charset=utf-8 meta tag (AMP).")
    if not soup.find('meta', {'name': 'viewport', 'content': re.compile(r'.*width=device-width.*')}):
        issues.append("Missing viewport meta tag with device-width (AMP).")
    amphtml = soup.find('link', rel='amphtml')
    if amphtml and amphtml['href'] != url:
        issues.append("amphtml link does not point to self (AMP).")
    return is_amp, issues

def check_pwa(base_url):
    """Check for PWA features"""
    issues = []
    manifest_found = False
    service_worker_found = False
    manifest_url = urljoin(base_url, '/manifest.json')
    try:
        response = requests.get(manifest_url, timeout=5)
        if response.status_code == 200:
            manifest = response.json()
            manifest_found = True
            if not manifest.get('name'):
                issues.append("Manifest missing 'name' field (PWA).")
            if not manifest.get('start_url'):
                issues.append("Manifest missing 'start_url' field (PWA).")
            if not manifest.get('display'):
                issues.append("Manifest missing 'display' field (PWA).")
    except Exception:
        issues.append("No manifest.json found (PWA).")
    sw_url = urljoin(base_url, '/service-worker.js')
    try:
        response = requests.head(sw_url, timeout=5)
        if response.status_code == 200:
            service_worker_found = True
    except:
        issues.append("No service worker found (PWA).")
    return manifest_found and service_worker_found, issues

def readability_analysis(text, language='en'):
    """Calculate Flesch-Kincaid Grade Level"""
    try:
        if language == 'en':
            return flesch_kincaid_grade(text)
        return None
    except Exception as e:
        logging.error(f"Error calculating readability: {e}")
        print(f"Error calculating readability: {e}")
        return None

def detect_language(text):
    """Detect primary language of content"""
    try:
        return langdetect.detect(text[:1000])
    except Exception as e:
        logging.warning(f"Error detecting language: {e}")
        return 'en'

def core_web_vitals(soup, load_time, ttfb):
    """Simulate Core Web Vitals with FCP estimation"""
    lcp_estimate = load_time
    fcp_estimate = ttfb + 0.5
    images = soup.find_all('img')
    for img in images[:5]:
        if img.get('src'):
            try:
                response = requests.head(img['src'], timeout=5)
                if 'content-length' in response.headers:
                    size_kb = int(response.headers['content-length']) / 1024
                    lcp_estimate += size_kb / 100
            except:
                pass
    cls_estimate = len(soup.find_all(['script', 'style'])) / 100
    return lcp_estimate, cls_estimate, fcp_estimate

def site_speed_optimization(soup, base_url):
    """Suggest site speed optimizations"""
    issues = []
    css_files = [link['href'] for link in soup.find_all('link', rel='stylesheet') if link.get('href')]
    js_files = [script['src'] for script in soup.find_all('script') if script.get('src')]
    for css in css_files[:5]:
        try:
            css_url = urljoin(base_url, css)
            response = requests.head(css_url, timeout=5)
            if 'content-length' in response.headers:
                size_kb = int(response.headers['content-length']) / 1024
                if size_kb > 50:
                    issues.append(f"CSS file {css} is {size_kb:.1f} KB; consider minification.")
            if not any(link.get('media') for link in soup.find_all('link', href=css)):
                issues.append(f"CSS file {css} is render-blocking; consider deferring or inlining critical CSS.")
        except:
            pass
    for js in js_files[:5]:
        try:
            js_url = urljoin(base_url, js)
            response = requests.head(js_url, timeout=5)
            if 'content-length' in response.headers:
                size_kb = int(response.headers['content-length']) / 1024
                if size_kb > 50:
                    issues.append(f"JS file {js} is {size_kb:.1f} KB; consider minification.")
            script = soup.find('script', src=js)
            if script and not script.get('defer') and not script.get('async'):
                issues.append(f"JS file {js} is render-blocking; consider adding defer or async.")
        except:
            pass
    images = soup.find_all('img')
    if images and not any(img.get('loading') == 'lazy' for img in images):
        issues.append("No images use lazy loading; consider adding 'loading=lazy'.")
    return issues

def crawl_budget_analysis(pages):
    """Suggest crawl budget optimizations"""
    low_value_pages = []
    for page in pages:
        soup = page['soup']
        text = page['text']
        word_count = len(re.findall(r'\b\w+\b', text))
        robots_meta = check_robots_meta(soup)
        if word_count < 200 or robots_meta['noindex']:
            low_value_pages.append((page['url'], 'Thin content' if word_count < 200 else 'Noindex'))
    return low_value_pages

def compute_tf_idf(text, all_texts):
    """TF-IDF with NLTK preprocessing"""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w.lower()) for w in word_tokenize(text) if w.lower() not in stop_words and w.isalnum()]
    tf = Counter(tokens)
    total_docs = len(all_texts) + 1
    idf = {word: math.log(total_docs / (1 + sum(1 for doc in all_texts if word in doc))) for word in tf}
    tf_idf = {word: tf[word] * idf.get(word, 1) for word in tf}
    return sorted(tf_idf.items(), key=lambda x: x[1], reverse=True)[:5]

def semantic_analysis(text, all_texts):
    """NLTK-based semantic analysis with LSI"""
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        texts = [text] + all_texts
        processed_texts = []
        for t in texts:
            tokens = [lemmatizer.lemmatize(w.lower()) for w in word_tokenize(t) if w.lower() not in stop_words and w.isalnum()]
            processed_texts.append(' '.join(tokens))
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(processed_texts)
        svd = TruncatedSVD(n_components=2)
        lsi_matrix = svd.fit_transform(tfidf_matrix)
        terms = vectorizer.get_feature_names_out()
        term_scores = svd.components_[0]
        top_indices = np.argsort(term_scores)[-5:]
        semantic_keywords = [terms[i] for i in top_indices]
        return semantic_keywords
    except Exception as e:
        logging.error(f"Error in semantic analysis: {e}")
        print(f"Error in semantic analysis: {e}")
        return []

def content_quality_score(text, soup, readability_score):
    """Calculate content quality score (0-100)"""
    score = 0
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w.lower()) for w in word_tokenize(text) if w.lower() not in stop_words and w.isalnum()]
    word_count = len(words)
    unique_words = len(set(words))
    sentences = sent_tokenize(text)
    avg_sentence_length = sum(len(word_tokenize(s)) for s in sentences) / max(1, len(sentences))
    keyword_density = sum(count for _, count in Counter(words).most_common(5)) / max(1, word_count)
    
    if word_count > 300:
        score += 30
    elif word_count > 150:
        score += 15
    if unique_words / max(1, word_count) > 0.5:
        score += 20
    if 10 < avg_sentence_length < 20:
        score += 20
    if readability_score and 6 <= readability_score <= 8:
        score += 20
    if 0.01 <= keyword_density <= 0.03:
        score += 10
    return score

def keyword_gap_analysis(target_text, competitor_text):
    """Keyword gap analysis with NLTK"""
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    target_words = set(lemmatizer.lemmatize(w.lower()) for w in word_tokenize(target_text) if w.lower() not in stop_words and w.isalnum())
    competitor_words = set(lemmatizer.lemmatize(w.lower()) for w in word_tokenize(competitor_text) if w.lower() not in stop_words and w.isalnum())
    return list(competitor_words - target_words)[:5]

def content_gap_analysis(target_soup, competitor_soup):
    """Analyze content gaps based on headings"""
    issues = []
    target_headings = [h.get_text(strip=True).lower() for h in target_soup.find_all(['h1', 'h2', 'h3'])]
    competitor_headings = [h.get_text(strip=True).lower() for h in competitor_soup.find_all(['h1', 'h2', 'h3'])]
    missing_headings = [h for h in competitor_headings if h not in target_headings]
    if missing_headings:
        issues.append(f"Missing {len(missing_headings)} heading topics: {', '.join(missing_headings[:3])}")
    return issues

def check_duplicate_metadata(pages):
    """Check for duplicate titles and meta descriptions"""
    duplicates = {'titles': [], 'meta_descs': []}
    title_map = {}
    desc_map = {}
    for page in pages:
        soup = page['soup']
        title = soup.find('title')
        title_text = title.text if title else ""
        meta_desc = soup.find('meta', attrs={'name': re.compile(r'description', re.I)})
        meta_desc_content = meta_desc['content'] if meta_desc and meta_desc.get('content') else ""
        if title_text:
            if title_text in title_map:
                duplicates['titles'].append((page['url'], title_map[title_text]))
            else:
                title_map[title_text] = page['url']
        if meta_desc_content:
            if meta_desc_content in desc_map:
                duplicates['meta_descs'].append((page['url'], desc_map[meta_desc_content]))
            else:
                desc_map[meta_desc_content] = page['url']
    return duplicates

def internal_link_optimization(anchor_texts):
    """Analyze internal link anchor text quality"""
    issues = []
    generic_anchors = ['click here', 'read more', 'learn more', 'here']
    anchor_counts = Counter(anchor.lower() for _, anchor in anchor_texts)
    for anchor, count in anchor_counts.items():
        if anchor in generic_anchors:
            issues.append(f"Generic anchor text '{anchor}' used {count} times; use descriptive anchors.")
        if count > 5:
            issues.append(f"Anchor text '{anchor}' used {count} times; avoid over-optimization.")
    if not anchor_texts:
        issues.append("No internal links found; add relevant internal links.")
    return issues

def check_duplicate_content(pages):
    """Check for duplicate content using cosine similarity"""
    if len(pages) < 2:
        return []
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    texts = []
    for page in pages:
        tokens = [lemmatizer.lemmatize(w.lower()) for w in word_tokenize(page['text']) if w.lower() not in stop_words and w.isalnum()]
        texts.append(' '.join(tokens))
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    similarities = cosine_similarity(tfidf_matrix)
    duplicates = []
    for i in range(len(pages)):
        for j in range(i + 1, len(pages)):
            if similarities[i][j] > 0.8:
                duplicates.append((pages[i]['url'], pages[j]['url'], similarities[i][j]))
    return duplicates

def social_shareability_score(soup, word_count):
    """Calculate social media shareability score"""
    score = 0
    og_tags, twitter_tags = check_social_media_tags(soup)
    if og_tags:
        score += 30
    if twitter_tags:
        score += 20
    if word_count > 300:
        score += 30
    images = soup.find_all('img')
    if len(images) > 0:
        score += 20
    return score

def internal_link_depth(pages):
    """Calculate internal link depth efficiently"""
    depth_map = {pages[0]['url']: 0}
    queue = [(pages[0]['url'], 0)]
    visited = set()
    while queue:
        url, depth = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        page = next((p for p in pages if p['url'] == url), None)
        if not page:
            continue
        internal_links, _, _ = extract_links(page['soup'], url)
        for link in internal_links:
            if link not in depth_map or depth + 1 < depth_map[link]:
                depth_map[link] = depth + 1
                queue.append((link, depth + 1))
    return depth_map

def export_to_csv(results):
    """Export SEO analysis to CSV"""
    try:
        with open('seo_analysis.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['URL', 'Title', 'Meta Description', 'Issues', 'Word Count', 'LCP', 'CLS', 'Content Quality', 'FCP'])
            for page in results:
                issues = page.get('issues', [])
                writer.writerow([
                    page['url'], page.get('title', ''), page.get('meta_desc', ''),
                    '; '.join(issues), page.get('word_count', 0),
                    page.get('lcp', 'N/A'), page.get('cls', 'N/A'),
                    page.get('content_quality', 'N/A'), page.get('fcp', 'N/A')
                ])
        print("Analysis exported to 'seo_analysis.csv'.")
    except Exception as e:
        logging.error(f"Error exporting to CSV: {e}")
        print(f"Error exporting to CSV: {e}")

def generate_dashboard(results):
    """Generate an HTML dashboard for SEO analysis"""
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>SEO Analysis Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .issue { color: red; }
        </style>
    </head>
    <body>
        <h1>SEO Analysis Dashboard</h1>
        <table>
            <tr>
                <th>URL</th>
                <th>Title</th>
                <th>Meta Description</th>
                <th>Word Count</th>
                <th>LCP (s)</th>
                <th>CLS</th>
                <th>FCP (s)</th>
                <th>Content Quality</th>
                <th>Issues</th>
            </tr>
    """
    for page in results:
        issues = '; '.join(page.get('issues', []))
        lcp = page.get('lcp', 'N/A')
        cls = page.get('cls', 'N/A')
        fcp = page.get('fcp', 'N/A')
        lcp_str = f"{lcp:.2f}" if isinstance(lcp, (int, float)) else lcp
        cls_str = f"{cls:.2f}" if isinstance(cls, (int, float)) else cls
        fcp_str = f"{fcp:.2f}" if isinstance(fcp, (int, float)) else fcp
        html += f"""
            <tr>
                <td>{page['url']}</td>
                <td>{page.get('title', '')}</td>
                <td>{page.get('meta_desc', '')}</td>
                <td>{page.get('word_count', 0)}</td>
                <td>{lcp_str}</td>
                <td>{cls_str}</td>
                <td>{fcp_str}</td>
                <td>{page.get('content_quality', 'N/A')}</td>
                <td class="issue">{issues}</td>
            </tr>
        """
    html += """
        </table>
    </body>
</html>
    """
    try:
        with open('seo_dashboard.html', 'w', encoding='utf-8') as f:
            f.write(html)
        print("Dashboard generated at 'seo_dashboard.html'.")
    except Exception as e:
        logging.error(f"Error generating dashboard: {e}")
        print(f"Error generating dashboard: {e}")

def generate_pdf_report(results, pages, url):
    """Generate a PDF report using LaTeX"""
    latex_content = r"""
    \documentclass{article}
    \usepackage{geometry}
    \usepackage{booktabs}
    \usepackage{hyperref}
    \usepackage{parskip}
    \usepackage{amsmath}
    \usepackage{graphicx}
    \usepackage{xcolor}
    \geometry{a4paper, margin=1in}
    \title{SEO Analysis Report for \texttt{\url}}
    \author{SEO Analyzer}
    \date{\today}
    \begin{document}
    \maketitle
    \section{Overview}
    This report provides a comprehensive SEO analysis for \texttt{\url}, covering technical SEO, content quality, and performance metrics. The analysis identified issues across pages, with recommendations to improve search engine rankings and user experience.

    \section{Summary}
    \begin{itemize}
        \item \textbf{Total Pages Analyzed}: \numberpages
        \item \textbf{Total Issues Found}: \totalissues
        \item \textbf{Average Content Quality Score}: \avgquality
        \item \textbf{Average Shareability Score}: \avgshare
    \end{itemize}

    \section{Detailed Analysis}
    \begin{table}[h]
        \centering
        \begin{tabular}{p{4cm}p{4cm}p{4cm}p{3cm}}
            \toprule
            \textbf{URL} & \textbf{Title} & \textbf{Issues} & \textbf{Content Quality} \\
            \midrule
            \tablerows
            \bottomrule
        \end{tabular}
        \caption{SEO Issues per Page}
    \end{table}

    \section{Recommendations}
    \begin{enumerate}
        \item \textbf{Optimize Titles and Meta Descriptions}: Ensure unique titles (50-60 chars) and meta descriptions (120-160 chars).
        \item \textbf{Improve Content Quality}: Aim for 300+ words, 6-8 readability score, and 1-3\% keyword density.
        \item \textbf{Enhance Site Speed}: Reduce LCP (<2.5s), CLS (<0.1), and FCP (<1.8s) by deferring render-blocking resources.
        \item \textbf{Add Structured Data}: Implement schema.org JSON-LD (e.g., Article, Product).
        \item \textbf{Fix Accessibility Issues}: Ensure alt text, captions, and proper heading structure.
    \end{enumerate}

    \section{Conclusion}
    Addressing the identified issues will improve the website's SEO performance, user engagement, and accessibility. Regular audits are recommended.

    \end{document}
    """
    total_issues = sum(len(result.get('issues', [])) for result in results)
    avg_quality = sum(result.get('content_quality', 0) for result in results) / max(1, len(results))
    # Calculate avg_share using pages list
    shareability_scores = []
    for result in results:
        page = next((p for p in pages if p['url'] == result['url']), None)
        if page and 'soup' in page:
            score = social_shareability_score(page['soup'], result['word_count'])
            shareability_scores.append(score)
    avg_share = sum(shareability_scores) / max(1, len(shareability_scores))
    table_rows = []
    for page in results:
        issues = '; '.join(page.get('issues', [])[:3]) + ('...' if len(page.get('issues', [])) > 3 else '')
        table_rows.append(f"{page['url']} & {page.get('title', '')[:30]} & {issues[:50]} & {page.get('content_quality', 'N/A')} \\\\")
    latex_content = latex_content.replace(r'\url', url) \
                                 .replace(r'\numberpages', str(len(results))) \
                                 .replace(r'\totalissues', str(total_issues)) \
                                 .replace(r'\avgquality', f"{avg_quality:.1f}/100") \
                                 .replace(r'\avgshare', f"{avg_share:.1f}/100") \
                                 .replace(r'\tablerows', '\n            '.join(table_rows))
    try:
        with open('seo_report.tex', 'w', encoding='utf-8') as f:
            f.write(latex_content)
        print("PDF report source generated at 'seo_report.tex'. Compile with latexmk to produce 'seo_report.pdf'.")
    except Exception as e:
        logging.error(f"Error generating PDF report: {e}")
        print(f"Error generating PDF report: {e}")

def crawl_website(url, max_pages=5):
    """Crawl website up to max_pages"""
    if not urlparse(url).scheme:
        url = 'https://' + url
    if not check_robots_txt(url):
        print("Scraping disallowed by robots.txt.")
        return []
    pages = []
    crawled_urls = set()
    to_crawl = [url]
    while to_crawl and len(pages) < max_pages:
        current_url = to_crawl.pop(0)
        if current_url in crawled_urls:
            continue
        crawled_urls.add(current_url)
        content, load_time, is_https, redirects, ttfb = get_page_content(current_url)
        if not content:
            continue
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text(strip=True).lower()
        word_count = len(word_tokenize(text))
        internal_links, external_links, anchor_texts = extract_links(soup, current_url)
        pages.append({
            'url': current_url,
            'soup': soup,
            'text': text,
            'load_time': load_time,
            'word_count': word_count,
            'is_https': is_https,
            'redirects': redirects,
            'external_links': external_links,
            'anchor_texts': anchor_texts,
            'ttfb': ttfb
        })
        to_crawl.extend([link for link in internal_links if link not in crawled_urls and len(to_crawl) < 20])
        time.sleep(1)
    return pages

def analyze_page(url, base_url, pages, competitor_text='', competitor_soup=None):
    """Analyze a single page for SEO"""
    content, load_time, is_https, redirects, ttfb = get_page_content(url, use_head=False)
    if not content:
        return None
    soup = BeautifulSoup(content, 'html.parser')
    text = soup.get_text(strip=True).lower()
    word_count = len(word_tokenize(text))
    language = detect_language(text)
    result = {
        'url': url,
        'text': text,
        'word_count': word_count,
        'lcp': None,
        'cls': None,
        'fcp': None,
        'content_quality': None,
        'issues': []
    }
    print(f"\nDetected Language: {language.upper()}")
    print("Recommendation: Tailor keywords and readability to the target audience's language.")
    title = soup.find('title')
    title_text = title.text if title else "No title found"
    title_length = len(title_text)
    print(f"\nTitle: {title_text}")
    print(f"Title Length: {title_length} characters")
    print("Recommendation: Keep title between 50-60 characters.")
    result['title'] = title_text
    if title_length < 50 or title_length > 60:
        result['issues'].append(f"Title length {title_length} is not optimal (50-60 characters).")
    meta_desc = soup.find('meta', attrs={'name': re.compile(r'description', re.I)})
    meta_desc_content = meta_desc['content'] if meta_desc and meta_desc.get('content') else "No meta description"
    meta_desc_length = len(meta_desc_content)
    print(f"\nMeta Description: {meta_desc_content}")
    print(f"Meta Description Length: {meta_desc_length} characters")
    print("Recommendation: Keep meta description between 120-160 characters.")
    result['meta_desc'] = meta_desc_content
    if meta_desc_length < 120 or meta_desc_length > 160:
        result['issues'].append(f"Meta description length {meta_desc_length} is not optimal (120-160 characters).")
    keywords = check_meta_keywords(soup)
    print(f"\nMeta Keywords: {keywords}")
    print("Recommendation: Use relevant meta keywords for search engines like Yandex.")
    if len(keywords) > 10:
        result['issues'].append("Too many meta keywords; avoid overuse.")
    headers = {'h1': [], 'h2': [], 'h3': []}
    for tag in ['h1', 'h2', 'h3']:
        headers[tag] = [h.get_text(strip=True) for h in soup.find_all(tag)]
    for tag, content in headers.items():
        print(f"\n{tag.upper()} Tags: {content}")
        print(f"Number of {tag.upper()}: {len(content)}")
    print("Recommendation: Use one H1 per page, organize with H2/H3.")
    if len(headers['h1']) != 1:
        result['issues'].append(f"Found {len(headers['h1'])} H1 tags; use exactly one.")
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w.lower()) for w in word_tokenize(text) if w.lower() not in stop_words and w.isalnum()]
    word_counts = Counter(words)
    top_keywords = word_counts.most_common(5)
    print("\nTop 5 Keywords by Frequency:")
    for word, count in top_keywords:
        print(f"{word}: {count} occurrences")
    print("Recommendation: Ensure primary keywords are relevant.")
    all_texts = [p['text'] for p in pages] + ([competitor_text] if competitor_text else [])
    tf_idf = compute_tf_idf(text, tuple(all_texts))
    print("\nSuggested Keywords (TF-IDF):")
    for word, score in tf_idf:
        print(f"{word}: {score:.3f}")
    print("Recommendation: Incorporate high-scoring keywords naturally.")
    semantic_keywords = semantic_analysis(text, all_texts)
    print("\nSemantic Keywords (LSI):")
    print(semantic_keywords)
    print("Recommendation: Include semantic keywords for better topical coverage.")
    if semantic_keywords:
        result['issues'].append(f"Consider adding semantic keywords: {', '.join(semantic_keywords)}")
    if competitor_text:
        missing_keywords = keyword_gap_analysis(text, competitor_text)
        print("\nMissing Keywords from Competitor Site:")
        print(f"Keywords on competitor site but not on page: {missing_keywords}")
        print("Recommendation: Consider adding these keywords.")
    if competitor_soup:
        content_gaps = content_gap_analysis(soup, competitor_soup)
        print("\nContent Gap Analysis:")
        for issue in content_gaps:
            print(f"- {issue}")
        result['issues'].extend(content_gaps)
    readability_score = readability_analysis(text, language=language)
    content_score = content_quality_score(text, soup, readability_score)
    print(f"\nContent Quality Score: {content_score:.1f}/100")
    print("Recommendation: Aim for a content quality score above 80.")
    result['content_quality'] = content_score
    if content_score < 80:
        result['issues'].append(f"Low content quality score: {content_score:.1f}/100")
    images = soup.find_all('img')
    missing_alt = [img for img in images if not img.get('alt')]
    print(f"\nImages without alt text: {len(missing_alt)}")
    print("Recommendation: Add descriptive alt text to all images.")
    if missing_alt:
        result['issues'].append(f"{len(missing_alt)} images lack alt text.")
    image_issues = check_image_optimization(soup, url)
    print(f"\nImage Optimization Issues: {len(image_issues)}")
    for issue in image_issues:
        print(f"- {issue}")
    result['issues'].extend(image_issues)
    speed_issues = site_speed_optimization(soup, url)
    print(f"\nSite Speed Issues: {len(speed_issues)}")
    for issue in speed_issues:
        print(f"- {issue}")
    result['issues'].extend(speed_issues)
    sitemap_urls = get_sitemap_urls(url)
    print(f"\nSitemap URLs: {sitemap_urls}")
    print(f"Number of Sitemap URLs: {len(sitemap_urls)}")
    print("Recommendation: Ensure a valid XML sitemap is submitted.")
    internal_links, external_links, anchor_texts = extract_links(soup, base_url)
    print(f"\nInternal Links: {len(internal_links)}")
    print(f"Sample Internal Links: {internal_links[:5]}")
    print(f"External Links: {len(external_links)}")
    print(f"Sample External Links: {external_links[:5]}")
    print("Recommendation: Use relevant internal links and high-quality external links.")
    link_optimization_issues = internal_link_optimization(anchor_texts)
    print(f"\nInternal Link Optimization Issues: {len(link_optimization_issues)}")
    for issue in link_optimization_issues:
        print(f"- {issue}")
    result['issues'].extend(link_optimization_issues)
    all_links = internal_links + external_links
    broken_links = check_broken_links(all_links)
    print(f"\nBroken Links: {len(broken_links)}")
    for link, status in broken_links:
        print(f"{link}: {status}")
    print("Recommendation: Fix or remove broken links.")
    if broken_links:
        result['issues'].append(f"{len(broken_links)} broken links detected.")
    redirect_chains = [(url, [r.url for r in redirects])] if redirects else []
    print(f"\nRedirect Chains: {len(redirect_chains)}")
    for page_url, chain in redirect_chains:
        print(f"{page_url} -> {' -> '.join(chain)}")
        if len(chain) > 2:
            result['issues'].append(f"Redirect chain for {page_url} has {len(chain)} hops; reduce to <=2.")
    print("Recommendation: Minimize redirect chains.")
    depth_map = internal_link_depth(pages)
    print(f"\nInternal Link Depth:")
    for page_url, depth in depth_map.items():
        print(f"{page_url}: Depth {depth}")
        if depth > 3:
            result['issues'].append(f"Page {page_url} is at depth {depth}; consider reducing to <=3.")
    print("Recommendation: Keep important pages within 3 clicks from homepage.")
    print(f"\nPage Load Time: {load_time:.2f} seconds")
    print("Recommendation: Aim for load time under 2 seconds.")
    if load_time > 2:
        result['issues'].append(f"Page load time {load_time:.2f}s is too high.")
    lcp, cls, fcp = core_web_vitals(soup, load_time, ttfb)
    print(f"\nEstimated Time to First Byte (TTFB): {ttfb:.2f} seconds")
    print(f"Estimated First Contentful Paint (FCP): {fcp:.2f} seconds")
    print(f"Estimated Largest Contentful Paint (LCP): {lcp:.2f} seconds")
    print(f"Estimated Cumulative Layout Shift (CLS): {cls:.2f}")
    print("Recommendation: Keep TTFB < 0.8s, FCP < 2.0s, LCP < 2.5s, CLS < 0.1.")
    result['lcp'] = lcp
    result['cls'] = cls
    result['fcp'] = fcp
    if ttfb > 0.8:
        result['issues'].append(f"TTFB {ttfb:.2f}s is too high.")
    if fcp > 2.0:
        result['issues'].append(f"FCP {fcp:.2f}s is too high.")
    if lcp > 2.5:
        result['issues'].append(f"LCP {lcp:.2f}s is too high.")
    if cls > 0.1:
        result['issues'].append(f"CLS {cls:.2f} is too high.")
    is_mobile_friendly = check_mobile_friendliness(soup)
    print(f"\nMobile-Friendly: {'Yes' if is_mobile_friendly else 'No'}")
    print("Recommendation: Ensure viewport meta tag.")
    if not is_mobile_friendly:
        result['issues'].append("Missing viewport meta tag for mobile-friendliness.")
    has_og, has_twitter = check_social_media_tags(soup)
    print(f"\nOpen Graph Tags Present: {'Yes' if has_og else 'No'}")
    print(f"Twitter Card Tags Present: {'Yes' if has_twitter else 'No'}")
    print("Recommendation: Add OG and Twitter Card tags.")
    if not has_og:
        result['issues'].append("Missing Open Graph tags.")
    if not has_twitter:
        result['issues'].append("Missing Twitter Card tags.")
    shareability_score = social_shareability_score(soup, word_count)
    print(f"\nSocial Media Shareability Score: {shareability_score}/100")
    print("Recommendation: Aim for a score above 80 for optimal social media engagement.")
    if shareability_score < 80:
        result['issues'].append(f"Low social shareability score: {shareability_score:.1f}/100")
    has_canonical = check_canonical_tag(soup, url)
    print(f"\nCanonical Tag Correct: {'Yes' if has_canonical else 'No'}")
    print("Recommendation: Ensure canonical tag points to correct URL.")
    if not has_canonical:
        result['issues'].append("Missing or incorrect canonical tag.")
    has_schema, schema_issues, schema_types = check_schema_markup(soup)
    print(f"\nSchema Markup Present: {'Yes' if has_schema else 'No'}")
    print(f"Schema Types: {schema_types}")
    if schema_issues:
        print("Schema Issues:", schema_issues)
    print("Recommendation: Add valid schema.org JSON-LD (e.g., Article, Product).")
    if not has_schema:
        result['issues'].append("Missing schema.org markup.")
        if 'article' in text.lower() and 'Article' not in schema_types:
            result['issues'].append("Consider adding Article schema for content type.")
    result['issues'].extend(schema_issues)
    has_breadcrumbs = check_breadcrumbs(soup)
    print(f"\nBreadcrumbs Present: {'Yes' if has_breadcrumbs else 'No'}")
    print("Recommendation: Add breadcrumb navigation for SEO and usability.")
    if not has_breadcrumbs:
        result['issues'].append("Missing breadcrumb navigation.")
    has_hreflang, hreflang_values = check_hreflang_tags(soup)
    print(f"\nHreflang Tags Present: {'Yes' if has_hreflang else 'No'}")
    print(f"Hreflang Values: {hreflang_values}")
    print("Recommendation: Use hreflang for multilingual sites.")
    if not has_hreflang:
        result['issues'].append("Missing hreflang tags for multilingual content.")
    js_dependent = check_js_dependency(soup)
    print(f"\nCritical Content JS-Dependent: {'Yes' if js_dependent else 'No'}")
    print("Recommendation: Ensure critical content is server-rendered.")
    if js_dependent:
        result['issues'].append("Critical content relies on JavaScript.")
    print(f"\nUses HTTPS: {'Yes' if is_https else 'No'}")
    print("Recommendation: Use HTTPS.")
    if not is_https:
        result['issues'].append("Site not using HTTPS.")
    robots_meta = check_robots_meta(soup)
    print(f"\nRobots Meta Tag - Noindex: {robots_meta['noindex']}")
    print(f"Robots Meta Tag - Nofollow: {robots_meta['nofollow']}")
    print("Recommendation: Avoid 'noindex' unless intentional.")
    if robots_meta['noindex']:
        result['issues'].append("Page has 'noindex' meta tag.")
    has_favicon = check_favicon(soup, url)
    print(f"\nFavicon Present: {'Yes' if has_favicon else 'No'}")
    print("Recommendation: Add a favicon for branding.")
    if not has_favicon:
        result['issues'].append("Missing favicon.")
    accessibility_issues = accessibility_audit(soup)
    print(f"\nAccessibility Issues: {len(accessibility_issues)}")
    for issue in accessibility_issues:
        print(f"- {issue}")
    result['issues'].extend(accessibility_issues)
    is_amp, amp_issues = check_amp(soup, url)
    print(f"\nAMP Page: {'Yes' if is_amp else 'No'}")
    if amp_issues:
        print("AMP Issues:", amp_issues)
    print("Recommendation: Ensure valid AMP markup for mobile performance.")
    result['issues'].extend(amp_issues)
    is_pwa, pwa_issues = check_pwa(base_url)
    print(f"\nPWA Supported: {'Yes' if is_pwa else 'No'}")
    if pwa_issues:
        print("PWA Issues:", pwa_issues)
    result['issues'].extend(pwa_issues)
    print("Recommendation: Implement PWA features for offline support.")
    if readability_score is not None:
        print(f"\nReadability (Flesch-Kincaid Grade Level): {readability_score:.2f}")
        print("Recommendation: Aim for a readability score of 6-8.")
        if readability_score > 8:
            result['issues'].append(f"Readability score {readability_score:.2f} is too high.")
    print(f"\nWord Count: {word_count}")
    print("Recommendation: Aim for 300+ words.")
    if word_count < 300:
        result['issues'].append(f"Word count {word_count} is too low.")
    return result

def analyze_seo(url, competitor_url=None):
    """Analyze SEO factors of a website"""
    pages = crawl_website(url)
    if not pages:
        print("No pages to analyze.")
        return None
    competitor_text = None
    competitor_soup = None
    if competitor_url:
        comp_content, _, _, _, _ = get_page_content(competitor_url)
        if comp_content:
            competitor_soup = BeautifulSoup(comp_content, 'html.parser')
            competitor_text = competitor_soup.get_text(strip=True).lower()
    analysis_results = []
    for page in pages:
        result = analyze_page(page['url'], url, pages, competitor_text, competitor_soup)
        if result:
            analysis_results.append(result)
    duplicates = check_duplicate_content(pages)
    print(f"\nPotential Duplicate Content: {len(duplicates)}")
    for url1, url2, similarity in duplicates:
        print(f"{url1} and {url2} (Similarity: {similarity:.2f})")
    print("Recommendation: Ensure unique content.")
    if duplicates:
        for result in analysis_results:
            if any(result['url'] in (url1, url2) for url1, url2, _ in duplicates):
                result['issues'].append("Potential duplicate content detected.")
    metadata_duplicates = check_duplicate_metadata(pages)
    print(f"\nDuplicate Titles: {len(metadata_duplicates['titles'])}")
    for url1, url2 in metadata_duplicates['titles']:
        print(f"Duplicate title between {url1} and {url2}")
    print(f"Duplicate Meta Descriptions: {len(metadata_duplicates['meta_descs'])}")
    for url1, url2 in metadata_duplicates['meta_descs']:
        print(f"Duplicate meta description between {url1} and {url2}")
    print("Recommendation: Ensure unique titles and meta descriptions.")
    if metadata_duplicates['titles']:
        for result in analysis_results:
            if any(result['url'] in (url1, url2) for url1, url2 in metadata_duplicates['titles']):
                result['issues'].append("Duplicate title detected.")
    if metadata_duplicates['meta_descs']:
        for result in analysis_results:
            if any(result['url'] in (url1, url2) for url1, url2 in metadata_duplicates['meta_descs']):
                result['issues'].append("Duplicate meta description detected.")
    low_value_pages = crawl_budget_analysis(pages)
    print(f"\nLow-Value Pages: {len(low_value_pages)}")
    for url, reason in low_value_pages:
        print(f"{url}: {reason}")
    print("Recommendation: Optimize or remove low-value pages.")
    referring_domains, domain_authority = analyze_backlinks_simulated(url, pages)
    if not referring_domains:
        print("Recommendation: Build high-quality backlinks to improve authority.")
    for result in analysis_results:
        if domain_authority < 10:
            result['issues'].append(f"Low estimated domain authority ({domain_authority}); consider backlink building.")
    sitemap_url = urljoin(url, '/sitemap.xml')
    submit_sitemap_simulated(sitemap_url, url)
    for page in pages:
        status = check_indexing_status_simulated(page['url'])
        if status != "Likely Indexed":
            for result in analysis_results:
                if result['url'] == page['url']:
                    result['issues'].append(f"Page may not be indexed: {status}")
    if analysis_results:
        export_to_csv(analysis_results)
        save_sitemap([page['url'] for page in pages])
        generate_dashboard(analysis_results)
        generate_pdf_report(analysis_results, pages, url)
        total_issues = sum(len(result['issues']) for result in analysis_results)
        avg_shareability = sum(social_shareability_score(page['soup'], page['word_count']) 
                              for page in pages) / len(pages)
        summary = f"SEO Analysis for {url}: {total_issues} issues found across {len(pages)} pages. Avg shareability: {avg_shareability:.1f}/100. #SEO #WebAnalysis"
        post_to_x_simulated(summary)
    return analysis_results

def main():
    url = input("Enter the website URL to analyze (e.g., https://example.com): ").strip()
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    competitor_url = input("Enter competitor URL for keyword gap analysis (or press Enter to skip): ").strip()
    if competitor_url and not competitor_url.startswith(('http://', 'https://')):
        competitor_url = 'https://' + competitor_url
    print(f"Analyzing {url}...")
    logging.info(f"Starting analysis for {url}")
    analyze_seo(url, competitor_url=competitor_url if competitor_url else None)
    logging.info(f"Completed analysis for {url}")
    print("Analysis complete. Check 'seo_analysis.csv', 'sitemap.xml', 'seo_dashboard.html', 'seo_report.tex', and 'seo_summary.txt' for results.")

if __name__ == "__main__":
    main()