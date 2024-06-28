LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY comparator  IS
	GENERIC (len : INTEGER := 8);   -- bit width out adder
	PORT (
		a, b         : IN  STD_LOGIC_VECTOR (len - 1 DOWNTO 0);
		gr        : OUT  STD_LOGIC      --- case of a >= b gr ='1' other wise 0
	);
END ENTITY comparator ;

ARCHITECTURE behavioral OF comparator  IS
BEGIN
	gr <= '1' WHEN a >= b ELSE '0';    --condition != STD_LOGIC_VECTOR (UNSIGNED(0)) ELSE '0';
END ARCHITECTURE behavioral;
