############### Task 1 ###############


class Cart:
    def __init__(self, items=[]):
        self.items = items if items else []

    def add(self, product):
        self.items.append(product)


cart1 = Cart()
cart2 = Cart()

cart1.add("phone")

print(cart2.items)
print(cart1.items)

############### Task 2 ###############
import asyncio


async def apply_discount(price, discount):
    if not discount:
        discount = 10

    response = httpx.get("https://promo-service.local/health")

    if response.status_code != 200:
        return price

    return price - discount


async def main():
    result = await asyncio.gather(
        apply_discount(100, 0),
        apply_discount(200, 20),
    )

    print(result)


asyncio.run(main())


############### Task 3 ###############


def retry(times):
    def wrapper(func):
        for _ in range(times):
            try:
                return func
            except Exception:
                pass

    return wrapper


@retry(3)
def load_user(user_id):
    if user_id <= 0:
        raise ValueError("Invalid user id")
    return {"id": user_id}


print(load_user(10))
