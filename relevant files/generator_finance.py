import random
from datetime import date, timedelta
from openpyxl import Workbook

def generate_account_excel(filename="account.xlsx"):
    # 创建 Excel 工作簿
    wb = Workbook()
    ws = wb.active
    ws.title = "流水"

    # 表头（无编号）
    ws.append(["项目", "日期", "金额"])

    # 日期范围
    start_date = date(2024, 1, 1)
    end_date = date(2025, 12, 31)

    # 收入 / 支出项目
    income_projects = [
        "客户回款", "项目结算", "技术服务费", "系统维护收入"
    ]
    expense_projects = [
        "员工工资", "办公用品", "房租", "水电费", "设备采购"
    ]

    current_date = start_date

    while current_date <= end_date:
        # 每天随机生成 0~3 条流水
        for _ in range(random.randint(0, 3)):
            if random.choice([True, False]):
                # 收入
                project = random.choice(income_projects)
                amount = round(random.uniform(1000, 20000), 2)
            else:
                # 支出
                project = random.choice(expense_projects)
                amount = -round(random.uniform(500, 15000), 2)

            ws.append([
                project,
                f"{current_date.year}.{current_date.month}.{current_date.day}",
                amount
            ])

        current_date += timedelta(days=1)

    # 保存 Excel 文件
    wb.save(filename)
    print(f"流水文件已生成：{filename}")

if __name__ == "__main__":
    generate_account_excel()
