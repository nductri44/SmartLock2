from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time

email_data = pd.read_excel("src/email_data.xlsx")

# Replace with your Gmail credentials
email_ids = list(email_data["sender_emails"])
passwords = list(email_data["passwords"])

# Replace with your email recipients
to_emails = list(email_data["recipent_emails"])

# count of send_emails
sent_emails = []

sent_emails_records = []

# Launch the web browser
driver = webdriver.Chrome()
driver.maximize_window()

# Loop through each Gmail account
for x in range(len(email_ids)):
    email_id = email_ids[x]
    password = passwords[x]

    sent_emails_records.extend(sent_emails)

    sent_emails.clear()

    # Navigate to gmail website
    driver.get('https://www.gmail.com')
    time.sleep(20)

    # Delete all cookies
    driver.delete_all_cookies()

    # Refresh the webpage to clear any data
    driver.refresh()
    time.sleep(10)

    # Enter email id and click Next
    driver.find_element(By.XPATH, '//input[@id="identifierId"]').send_keys(email_id)
    driver.find_element(By.XPATH, '//span[contains(text(), "Next")]').click()
    time.sleep(10)

    # Enter password and click Next
    driver.find_element(By.XPATH, '//input[@type="password"]').send_keys(email_id)
    driver.find_element(By.XPATH, '//span[contains(text(), "Next")]').click()
    time.sleep(10)

    # Loop through recipient email id
    for i in range(len(to_emails)):
    # Get the current chunk of 5 elements
        to = to_emails[i]
        if to not in sent_emails_records:
            # Compose email
            driver.find_element(By.XPATH, '//input[@class="agP aFw"]').send_keys(to)
            time.sleep(10)
            driver.find_element(By.XPATH, '//input[@name="subjectbox"]').send_keys('Subject of the email')
            time.sleep(10)
            driver.find_element(By.XPATH, '//div[@aria-label="Message Body"]').send_keys('Body of the email')

            # Send the email
            driver.find_element(By.XPATH, '//div[@class="T-I J-J5-Ji aoO v7 T-I-atl L3"]').click()
            time.sleep(10)

            print("Email sent to : ", to)

            sent_emails.append(to)

            if len(sent_emails) >= 2:
                print("Logging of current email")

                break

            else:

                continue

    # Logout of gmail:
    driver.get("https://accounts.google.com/Logout?hl=en")
    time.sleep(10)