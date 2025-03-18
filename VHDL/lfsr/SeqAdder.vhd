-- MIT License

-- Copyright (c) 2024 m-spr

-- Permission is hereby granted, free of charge, to any person obtaining a copy
-- of this software and associated documentation files (the "Software"), to deal
-- in the Software without restriction, including without limitation the rights
-- to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
-- copies of the Software, and to permit persons to whom the Software is
-- furnished to do so, subject to the following conditions:

-- The above copyright notice and this permission notice shall be included in all
-- copies or substantial portions of the Software.

-- THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
-- IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
-- FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
-- AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
-- LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
-- OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
-- SOFTWARE.

LIBRARY IEEE;
USE IEEE.STD_LOGIC_1164.ALL;
USE IEEE.NUMERIC_STD.ALL;

ENTITY SeqAdder IS
    GENERIC (
        lenPop : INTEGER := 8  -- Bit width of popCounters
    );
    PORT (
        clk         : IN  STD_LOGIC;
        rst         : IN  STD_LOGIC;
        reg1Update  : IN  STD_LOGIC;
        reg1rst     : IN  STD_LOGIC;
        reg2Update  : IN  STD_LOGIC;
        reg2rst     : IN  STD_LOGIC;
        din         : IN  STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0);
        dout        : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
    );
END ENTITY SeqAdder;

ARCHITECTURE behavioral OF SeqAdder IS
    SIGNAL addToReg, addToAdder : STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0);

    COMPONENT reg IS
        GENERIC (
            lenPop : INTEGER := 8  -- Bit width of popCounters
        );
        PORT (
            clk       : IN  STD_LOGIC;
            regUpdate : IN  STD_LOGIC;
            regrst    : IN  STD_LOGIC;
            din       : IN  STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0);
            dout      : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
        );
    END COMPONENT;

BEGIN

    reg1 : reg 
        GENERIC MAP (
            lenPop => lenPop
        )
        PORT MAP (
            clk       => clk,
            regUpdate => reg1Update,
            regrst    => reg1rst,
            din       => addToReg,
            dout      => addToAdder
        );

    reg2 : reg 
        GENERIC MAP (
            lenPop => lenPop
        )
        PORT MAP (
            clk       => clk,
            regUpdate => reg2Update,
            regrst    => reg2rst,
            din       => addToAdder,
            dout      => dout
        );

    -- Addition operation using unsigned arithmetic
    addToReg <= STD_LOGIC_VECTOR(UNSIGNED(din) + UNSIGNED(addToAdder));

END ARCHITECTURE behavioral;
