import scrapy

class NewsArticleSpider(scrapy.Spider):
    
    name = 'news_article_spider'
    
    def _init_(self, url):
        self.start_url = url
    
    # start_urls = [some url]
    
    def parse(self, response):
        
        # Scrape huffpost
        if response.url.startswith('https://www.huffpost.com/'):
            
            # Extract headline
            headline = response.css('h1.headline::text').get()
            
            # Extract text
            text = response.css('div.entry__content-and-right-rail-container '
                                'div.primary-cli.cli.cli-text p::text, '
                                'div.primary-cli.cli.cli-text a::text, '
                                'div.primary-cli.cli.cli-text p span::text, '
                                'div.primary-cli.cli.cli-text a span::text').getall()
            text = ' '.join(text)
            
            # Output
            yield {
                'link': response.url,
                'headline' : headline,
                'text':text
            }
        
        # Scrape New York Times
        elif response.url.startswith('https://www.nytimes.com/'):
            
            # Extract headline
            headline = response.css('h1[class$="e1h9rw200"]::text').get()
            
            # Extract text
            text = response.css('div.css-53u6y8 p::text, '
                                'div.css-53u6y8 a::text, '
                                'div.entry-content p.story-body-text::text, '
                                'div.entry-content a::text, '
                                'div.entry-content li::text').getall()
            text = ' '.join(text)
            
            # Output
            yield {
                'link': response.url,
                'headline': headline,
                'text':text
            }