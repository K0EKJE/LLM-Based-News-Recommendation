from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from multiprocessing import Process, Queue
import scrapy
from news_articles.news_articles.spiders.news_spider import NewsArticleSpider

def run_spider_process(queue, start_url):
    try:
        results = []

        def spider_results(signal, sender, item, response, spider):
            results.append(item['text'])

        process = CrawlerProcess(get_project_settings())
        process.crawl(NewsArticleSpider, start_url=start_url)

        for p in process.crawlers:
            p.signals.connect(spider_results, signal=scrapy.signals.item_scraped)

        process.start()
        queue.put(results)
    except Exception as e:
        queue.put(e)

class NewsArticleSpiderRunner:
    @staticmethod
    def run_spider(start_url):
        queue = Queue()
        process = Process(target=run_spider_process, args=(queue, start_url))
        process.start()
        result = queue.get()
        process.join()

        if isinstance(result, Exception):
            raise result

        return result

# import scrapy
# from scrapy.crawler import CrawlerProcess
# from news_articles.news_articles.spiders.news_spider import NewsArticleSpider
# from scrapy.utils.project import get_project_settings
#
# class NewsArticleSpiderRunner:
#     @staticmethod
#     def run_spider(url):
#         def spider_results(signal, sender, item, response, spider):
#             results.append(item[0])
#
#         results = []
#         process = CrawlerProcess(get_project_settings())
#         process.crawl(NewsArticleSpider, start_url=url)
#
#         for p in process.crawlers:
#             p.signals.connect(spider_results, signal=scrapy.signals.item_scraped)
#
#         process.start()  # the script will block here until the crawling is finished
#
#         return results
