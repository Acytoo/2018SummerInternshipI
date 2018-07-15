import requests
from bs4 import BeautifulSoup
import os

'''
得到给定网址内的老虎图片的url,返回list
'''

def get_tiger_pics_urls(url_tiger):
	# 构造请求头
	head={}
	head['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64; rv:45.0) Gecko/20100101 Firefox/45.0'
	# 防止请求超时
	ret_urls = []
	try :
		response = requests.get(url_tiger, headers = head)
		soup = BeautifulSoup(response.text, 'html.parser')
		i = 0
		for img_url in soup.find_all('a', attrs={'class':'pic'}):
			ret_urls.append(img_url.find('img').attrs['src'])
			print(i)
			i = i + 1
	except:
		print('请求超时。。。 ，跳过')

	return ret_urls


def get_tiger_pics(url_list):
	head={}
	head['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64; rv:45.0) Gecko/20100101 Firefox/45.0'
	for url in url_list:
		try:
			response = requests.get(url, headers = head)
			with open('tigers/' + url.split('/')[-1], 'wb') as f:
				f.write(response.content)
		except:
			pass


if __name__ == '__main__':

	# first mkdir the dir to store the pics
	if (os.path.exists('tigers')):
		pass
	else:
		os.makedirs('tigers')

	# then we can find all the urls that link to the pic to the jpg
		
	original_url_list = ['http://www.tooopen.com/img/89_873.aspx', 
						 'http://www.tooopen.com/img/89_873_1_2.aspx',
						 'http://www.tooopen.com/img/89_873_1_3.aspx',
						 'http://www.tooopen.com/img/89_873_1_4.aspx',
						 'http://www.tooopen.com/img/89_873_1_5.aspx',
						 'http://www.tooopen.com/img/89_873_1_6.aspx']
	list_url = []

	for original_url in original_url_list:
		list_url.extend(get_tiger_pics_urls(original_url))

	# finally we can get pics and store them
	print('start saving pics')
	get_tiger_pics(list_url)

	print('done')
		


