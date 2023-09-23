import scrapy
from scrapy.http import Request
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import regex as re

class LangcrawlItem(scrapy.Item):
    # define the fields for your item here like:
    word = scrapy.Field()
    pronunciation = scrapy.Field()

# Note: Need to initialize name and start_url list in the subclasses. 
class LangSpider(CrawlSpider):
    allowed_domains = ["en.wiktionary.org"]

    rules = [ 
        Rule(
            LinkExtractor(
                restrict_xpaths = ('//div[@id="mw-pages"]'),
                tags = ('a'),
                unique = True,
            ),
            callback = "parse_item",
            process_request = "filter_request",
            follow = True
        )
    ]
    def filter_request(self, request, response):
        url : str = request.url

        if url in self.start_urls:
            return request
        
        # For clicking the next button
        if url.find("index.php") > 0 and url.find('pagefrom=') > 0:
            return request
        
        # For scraping the words themselves
        if url[7:].find(":") < 0:
            return request

        return None

    def parse_item(self, response):
        if (response.url in self.start_urls):
            return
        
        # Visit next page
        if response.url.find("index.php") > 0:
            print(response.url)
            yield scrapy.Request(url = response.url)

        # Perform scraping
        else:
            item = LangcrawlItem()
            item["word"] = response.xpath('//title/text()').extract()[0].removesuffix("- Wiktionary, the free dictionary").strip().lower()
            
            responses =response.xpath(f'//h2/span[@class="mw-headline"]/text() | //span[@class="IPA"]/text()').extract()

            filtered_responses = []
            # Extract only the responses in the language
            is_in_scope = False
            
            for x in responses:
                if is_in_scope is True:
                    if "/" in x or "[" in x:
                        filtered_responses.append(x)
                    else: 
                        break
                    
                if x == self.name:
                    is_in_scope = True 

            # Seelct only the responses that correspond to the language.
            item["pronunciation"] = ','.join(filtered_responses)
            
            yield item 