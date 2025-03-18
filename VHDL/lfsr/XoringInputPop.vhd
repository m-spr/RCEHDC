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

ENTITY XoringInputPop IS
    GENERIC (
        n : INTEGER := 8  -- Number of loop iterations
    );     
    PORT (
        clk      : IN  STD_LOGIC;
        rst      : IN  STD_LOGIC;
        update   : IN  STD_LOGIC;  -- Run should be '1' for 2 clock cycles (to be considered later)
        done     : IN  STD_LOGIC;
        din      : IN  STD_LOGIC;
        BV       : IN  STD_LOGIC;
        dout     : OUT STD_LOGIC_VECTOR(n-1 DOWNTO 0)
    );
END ENTITY XoringInputPop;

ARCHITECTURE behavioral OF XoringInputPop IS

    -- Pop Count Component Declaration
    COMPONENT popCount IS
        GENERIC (
            lenPop : INTEGER := 8  -- Bit width of output popCounters (LOG2(#feature))
        );
        PORT (
            clk  : IN  STD_LOGIC;
            rst  : IN  STD_LOGIC;
            en   : IN  STD_LOGIC;
            dout : OUT STD_LOGIC_VECTOR(lenPop-1 DOWNTO 0)
        );
    END COMPONENT;

    -- Register Component Declaration
    COMPONENT reg IS
        GENERIC (
            lenPop : INTEGER := 8  -- Bit width of output popCounters
        );
        PORT (
            clk      : IN  STD_LOGIC;
            regUpdate : IN  STD_LOGIC;
            regrst    : IN  STD_LOGIC;
            din      : IN  STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0);
            dout     : OUT STD_LOGIC_VECTOR(lenPop - 1 DOWNTO 0)
        );
    END COMPONENT;

    -- Internal Signals
    SIGNAL XORR   : STD_LOGIC;
    SIGNAL poprst : STD_LOGIC;
    SIGNAL doutI  : STD_LOGIC_VECTOR(n-1 DOWNTO 0);

BEGIN

    -- Pop Count Instance with Explicit Generic Mapping
    pop : popCount
        GENERIC MAP (
            lenPop => n  -- Bit width of popCounters
        )
        PORT MAP (
            clk  => clk,
            rst  => poprst,
            en   => XORR,
            dout => doutI
        );

    -- XOR Logic and Reset Signal
    XORR   <= (din XNOR BV) AND update;
    poprst <= rst OR done;

    -- Register Instance with Explicit Generic Mapping
    outreg : reg 
        GENERIC MAP (
            lenPop => n  -- Bit width of popCounters
        )
        PORT MAP (
            clk      => clk,
            regUpdate => done,
            regrst    => rst,
            din      => doutI,
            dout     => dout
        );

END ARCHITECTURE behavioral;
