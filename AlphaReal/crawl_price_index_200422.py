import time
import sys
import traceback
import psycopg2 as pg
import zipfile
import csv
import os

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import smtplib

_logFilePath = os.path.abspath("log/log_population.txt")
_tempFolder = os.path.abspath("temp")
_csvFileName = _tempFolder + "/insert.csv"
_chromeDriverPath = os.path.abspath("dependency/chromedriver")


_f = open(_logFilePath, "a")


def logging(txt):
    timestr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

    print("[" + timestr + "] {}".format(txt))
    _f.write("[" + timestr + "] {}\n".format(txt))


def emptyTempFolder():
    path = os.path.abspath(_tempFolder)

    for file in os.listdir(path):
        os.remove(path + "/" + file)


def crwaling():
    options = Options()

    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    prefs = {"download.default_directory": _tempFolder}  # 다운로드 경로 설정
    options.add_experimental_option("prefs", prefs)

    driver = webdriver.Chrome(executable_path=_chromeDriverPath, options=options)

    # kosis에 접속
    logging("페이지에 접속합니다 ...")
    driver.get(
        "http://kosis.kr/statisticsList/statisticsListIndex.do?menuId=M_01_01&vwcd=MT_ZTITLE&parmTabId=M_01_01#SelectStatsBoxDiv"
    )

    wait = WebDriverWait(driver, 10)

    # 인구.가구 메뉴로 이동
    logging("인구.가구 메뉴로 이동합니다 ...")
    element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#A\\.1 > a")))
    element.click()

    # 주민등록인구현황 메뉴로 이동
    logging("주민등록인구현황 메뉴로 이동합니다 ...")
    element = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "#A6\\.2 > a")))
    element.click()

    # 행정구역(읍면동)별/5세별 주민등록인구 다운로드 메뉴로 이동
    logging("행정구역(읍면동)별/5세별 주민등록인구 다운로드 메뉴로 이동합니다 ...")
    # print(driver.page_source)
    # print('--------------------------------------------------------------')
    element = wait.until(
        EC.element_to_be_clickable((By.XPATH, '//*[@id="A6.2"]/ul/li[4]/button[1]'))
    )
    driver.execute_script(
        "document.evaluate('//*[@id=\"A6.2\"]/ul/li[4]/button[1]', document, null, XPathResult.FIRST_ORDERED_NODE_TYPE, null).singleNodeValue.click()"
    )

    # 통계표 파일 서비스 메뉴로 이동
    logging("통계표 파일 서비스 메뉴로 이동합니다 ...")
    # print('--------------------------------------------------------------')
    # print(driver.page_source)
    # wait.until(EC.frame_to_be_available_and_switch_to_it(driver.find_element_by_css_selector("#filesvcLayer_101_DT_1B04005N > iframe")))
    time.sleep(10)
    iframe = driver.switch_to.frame(
        driver.find_element_by_css_selector("#filesvcLayer_101_DT_1B04005N > iframe")
    )

    logging("통계표 파일 목록을 가져옵니다 ...")
    element = wait.until(
        EC.element_to_be_clickable(
            (
                By.CSS_SELECTOR,
                "body > table > tbody > tr:nth-child(2) > td > table > tbody > tr:nth-child(4) > td > table > tbody > tr:nth-child(1) > td.checkbox > a > img",
            )
        )
    )
    trs = driver.find_element_by_css_selector(
        "body > table > tbody > tr:nth-child(2) > td > table > tbody > tr:nth-child(4) > td > table > tbody"
    ).find_elements_by_tag_name("tr")

    # 각각의 목록 순회하면서 파일 다운로드 받기
    for idx in range(len(trs)):
        logging("데이터를 가져오는 중입니다... [%i/%i]" % (idx, len(trs)))

        tr = trs[idx]

        tr.find_element_by_css_selector("td.checkbox > a").click()

        wait = True

        while wait:
            for fname in os.listdir(_tempFolder):
                if "crdownload" in fname:
                    logging("파일을 다운로드 중입니다 ...")
                    time.sleep(3)
                else:
                    wait = False

        logging("파일 다운로드가 완료되었습니다 ...")

        # 받아온 파일 압축 풀기
        tempFolder = os.path.abspath(_tempFolder)
        zipfileName = tempFolder + "/" + os.listdir(tempFolder)[0]  # 압축파일명 가져오기
        zip = zipfile.ZipFile(zipfileName, "r")  # 압축 객체 만들기
        zip.extractall(path=tempFolder)  # 압축 풀기
        zip.close()  # 메모리 해제
        os.remove(zipfileName)  # 압축파일 삭제하기

        # csv 포맷 DB에 맞춰 정리하기
        csvReadfileName = tempFolder + "/" + os.listdir(tempFolder)[0]
        csvReadfile = open(csvReadfileName, "r", encoding="EUC-KR")
        reader = csv.reader(csvReadfile)
        rows = []

        idx = 0

        for row in reader:
            # 첫 3줄 제거
            if idx > 3:
                for idx2 in range(len(row)):
                    if idx2 == 0:
                        code = row[idx2].replace("'", "")

                        # 읍면동의 경우
                        if len(code) == 10:
                            # row.insert(1, '"%s"' % code)
                            row.insert(1, code)
                            row.insert(1, code[0:5])
                            row.insert(1, code[0:2])
                        # 시군구의 경우
                        elif len(code) == 5:
                            row.insert(1, "")
                            row.insert(1, code)
                            row.insert(1, code[0:2])
                        # 시도의 경우
                        elif len(code) == 2:
                            row.insert(1, "")
                            row.insert(1, "")

                            if code == "00":
                                code = 0

                            row.insert(1, code)

                        del row[0]

                    # elif idx == 3 or idx == 5:
                    # row[idx2].replace('"', '')

                rows.append(row)
            idx += 1

        csvReadfile.close()

        # 테이블에 넣을 csv 파일 생성하기
        csvWriteFileName = open(_csvFileName, "w", encoding="EUC-KR")
        writer = csv.writer(csvWriteFileName)

        for row in rows:
            writer.writerow(row)

        csvWriteFileName.close()

        # insert.csv파일 권한 변경
        # os.chmod(_csvFileName, 0o777)

        # 테이블에 데이터 넣기

        # temp 폴더 비우기
        emptyTempFolder()
    # chromedriver 닫기
    logging("chorme driver을 종료합니다 ...")
    driver.close()


def main():
    # temp 폴더 비우기
    logging("임시 폴더를 초기화합니다 ...")
    emptyTempFolder()

    # 크롤링 시작하기
    logging("크롤링을 시작합니다 ...")
    crwaling()
    logging("크롤링을 종료합니다 ...")


if __name__ == "__main__":
    main()
    # quit()
