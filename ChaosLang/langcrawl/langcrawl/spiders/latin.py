import scrapy
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
import regex as re

class LangcrawlItem(scrapy.Item):
    # define the fields for your item here like:
    word = scrapy.Field()
    pronunciation = scrapy.Field()


class LatinSpider(CrawlSpider):
    name = "latin"
    allowed_domains = ["en.wiktionary.org"]

    # https://en.wiktionary.org/wiki/Category:Latin_terms_with_IPA_pronunciation
    # https://en.wiktionary.org/wiki/Iabes#Latin
    start_urls = ["https://en.wiktionary.org/wiki/Category:Latin_terms_with_IPA_pronunciation"]

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
            yield scrapy.Request(url = response.url)

        # Perform scraping
        else:
            item = LangcrawlItem()
            item["word"] = response.xpath('//title/text()').extract()[0].removesuffix("- Wiktionary, the free dictionary").strip().lower()
            item["pronunciation"] = response.xpath('//span[@class="IPA"]/text()').extract()

            print(item['word'])
            yield item 
    

